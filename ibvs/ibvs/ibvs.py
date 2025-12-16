#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import cv2
import math
import pyrealsense2 as rs

from pupil_apriltags import Detector

from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Header

# ============================================================
# ================= USER-EDITABLE PARAMETERS =================
# ============================================================

# -------- Camera intrinsics (EDIT THESE) --------
FX = 615.0
FY = 615.0
CX = 320.0
CY = 240.0

# -------- AprilTag --------
TAG_SIZE = 0.25  # meters
TARGET_TAG_ID = 1
DESIRED_SIZE = 180
EPS = 1e-6

# -------- IBVS --------
LAMBDA_P = 0.6
PATCH = 3
Z_DES = 1.0  # desired distance (meters)

# -------- UDP video stream --------
QGC_IP = "192.168.4.1"
QGC_PORT = 5600

# -------- Camera → Body transform --------
R_CB = np.eye(3)

P_CB = np.array([
    [-0.2],
    [ 0.0],
    [ 0.0]
])

# ============================================================


def interaction_matrix(u, v, Z, fx, fy, cx, cy):
    x = (u - cx) / fx
    y = (v - cy) / fy

    L = np.array([
        [-1.0/Z,     0.0,    x / Z,    x * y,    -(1 + x*x),   y],
        [0.0,     -1.0/Z,    y / Z,    1 + y*y,  -x * y,      -x],
        [0.0,        0.0,   -1.0,     -y * Z,     x * Z,      0.0]
    ])

    return L


def build_IBVS_matrix(pts, desired, depth_img, fx, fy, cx, cy, Z_des):
    rows = []
    errs = []
    h, w = depth_img.shape

    for i in range(4):
        u, v = pts[i]
        ui, vi = int(round(u)), int(round(v))

        if ui < 0 or ui >= w or vi < 0 or vi >= h:
            return None, None, False, None

        patch = depth_img[
            max(0, vi-PATCH):min(h, vi+PATCH+1),
            max(0, ui-PATCH):min(w, ui+PATCH+1)
        ]
        valid = patch[patch > 0]
        if valid.size == 0:
            return None, None, False, None

        Z = float(np.median(valid))
        if Z <= 0 or np.isnan(Z):
            return None, None, False, None

        L = interaction_matrix(u, v, Z, fx, fy, cx, cy)

        rows.append(L[0])
        rows.append(L[1])
        rows.append(L[2])

        errs.append([u - desired[i, 0]])
        errs.append([v - desired[i, 1]])
        errs.append([Z - Z_des])

    Ls = np.vstack(rows)
    e  = np.vstack(errs)

    return Ls, e, True, Z


# ============================================================


class IBVSAprilTagNode(Node):

    def __init__(self):
        super().__init__("ibvs_apriltag_pnp")

        # ---------------- Publishers ----------------
        self.vision_pub = self.create_publisher(
            PoseStamped,
            "/mavros/vision_pose/pose",
            10
        )

        self.vel_pub = self.create_publisher(
            Twist,
            "/mavros/setpoint_velocity/cmd_vel_unstamped",
            10
        )

        # ---------------- RealSense ----------------
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = self.pipeline.start(cfg)

        self.align = rs.align(rs.stream.color)

        # ---------------- Camera model ----------------
        self.fx, self.fy, self.cx, self.cy = FX, FY, CX, CY

        self.camera_matrix = np.array([
            [FX, 0, CX],
            [0, FY, CY],
            [0,  0,  1]
        ], dtype=np.float32)

        self.dist_coeffs = np.zeros((4,1))

        # ---------------- AprilTag ----------------
        self.detector = Detector(
            families="tag36h11",
            nthreads=2,
            quad_decimate=1.0,
            refine_edges=True
        )

        s = TAG_SIZE / 2.0
        self.object_points = np.array([
            [-s, -s, 0],
            [ s, -s, 0],
            [ s,  s, 0],
            [-s,  s, 0],
        ], dtype=np.float32)

        # Desired image feature (centered square)
        self.desired = np.array([
    [320-DESIRED_SIZE//2, 240-DESIRED_SIZE//2],
    [320+DESIRED_SIZE//2, 240-DESIRED_SIZE//2],
    [320+DESIRED_SIZE//2, 240+DESIRED_SIZE//2],
    [320-DESIRED_SIZE//2, 240+DESIRED_SIZE//2]
        ], dtype=np.float32)

        # ---------------- UDP Video ----------------
        self.video = cv2.VideoWriter(
            f"appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=ultrafast "
            f"! rtph264pay config-interval=1 pt=96 "
            f"! udpsink host={QGC_IP} port={QGC_PORT}",
            cv2.CAP_GSTREAMER,
            0,
            30,
            (640, 480),
            True
        )

        self.timer = self.create_timer(0.05, self.loop)
        self.get_logger().info("IBVS + PnP unified node started")

    # ============================================================
    def loop(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * 0.001

        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)

        if len(detections) == 0:
            self.video.write(color)
            return

        tag = detections[0]
        pts = np.array(tag.corners, dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            self.object_points,
            pts,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if not success:
            return

        # ---------------- Pose publish ----------------
        z_cam = float(tvec[2][0])
        x_cam = float(tvec[0][0])
        y_cam = float(tvec[1][0])

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "map"
        pose.pose.position.x = z_cam
        pose.pose.position.y = x_cam
        pose.pose.position.z = y_cam

        R,_ = cv2.Rodrigues(rvec)
        yaw = math.atan2(R[1,0], R[0,0])
        pitch = math.asin(-R[2,0])
        roll = math.atan2(R[2,1], R[2,2])

        q = self.euler_to_quaternion(roll, pitch, yaw)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        self.vision_pub.publish(pose)

        # ---------------- IBVS ----------------
        Ls, e, ok, _ = build_IBVS_matrix(
            pts, self.desired, depth,
            self.fx, self.fy, self.cx, self.cy, Z_DES
        )

        if not ok:
            return

        Vc = -LAMBDA_P * np.linalg.pinv(Ls) @ e

        v_c = Vc[0:3,0].reshape(3,1)
        w_c = Vc[3:6,0].reshape(3,1)

        w_b = R_CB @ w_c
        v_b = R_CB @ v_c + np.cross(
            w_b.flatten(),
            P_CB.flatten()
        ).reshape(3,1)

        # ---------------- Publish velocity ----------------
        cmd = Twist()
        cmd.linear.x  = float(v_b[0])
        cmd.linear.y  = float(v_b[1])
        cmd.linear.z  = float(v_b[2])
        cmd.angular.x = float(w_b[0])
        cmd.angular.y = float(w_b[1])
        cmd.angular.z = float(w_b[2])

        self.vel_pub.publish(cmd)

        # ---------------- Draw + Stream ----------------
        for i,(u,v) in enumerate(pts):
            cv2.circle(color,(int(u),int(v)),5,(255,0,0),-1)
            cv2.putText(color,f"{i+1}",(int(u)+6,int(v)-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

	# ---------------- Draw detected tag (GREEN) ----------------
        for i in range(4):
            p1 = tuple(pts[i].astype(int))
            p2 = tuple(pts[(i+1) % 4].astype(int))
            cv2.line(color, p1, p2, (0, 255, 0), 2)
            cv2.circle(color, p1, 4, (255, 0, 0), -1)

	# ---------------- Draw desired IBVS target (RED) ----------------
        for i in range(4):
           d1 = tuple(self.desired[i].astype(int))
           d2 = tuple(self.desired[(i+1) % 4].astype(int))
           cv2.line(color, d1, d2, (0, 0, 255), 2)


        self.video.write(color)

    # ============================================================
    def euler_to_quaternion(self, roll, pitch, yaw):
        cy = math.cos(yaw*0.5)
        sy = math.sin(yaw*0.5)
        cp = math.cos(pitch*0.5)
        sp = math.sin(pitch*0.5)
        cr = math.cos(roll*0.5)
        sr = math.sin(roll*0.5)

        return (
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy,
            cr*cp*cy + sr*sp*sy
        )


def main():
    rclpy.init()
    node = IBVSAprilTagNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

