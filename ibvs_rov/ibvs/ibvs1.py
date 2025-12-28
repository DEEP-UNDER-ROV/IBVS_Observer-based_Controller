# IBVS Integral Velocity -> Normal LS no Yaw

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import PolygonStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ibvs.constants import *


class IBVSControllerNode(Node):
    def __init__(self):
        super().__init__("IBVSControllerNode")
        self.bridge = CvBridge()

        # ---------------- Subscribers ----------------
        self.sub_corners = self.create_subscription(
            PolygonStamped, "/apriltag/corners", self.cb_corners, 10
        )
        self.sub_depth = self.create_subscription(
            Image, "/camera/depth/image_raw", self.cb_depth, 10
        )
        self.sub_pnp = self.create_subscription(
            Odometry, "/pnp/relative_position", self.cb_pnp, 10
        )

        # ---------------- Publishers -----------------
        self.odom_pub = self.create_publisher(
            Odometry, "/mavros/odometry/in", 10
        )

        self.err_pub = self.create_publisher(
            Float32MultiArray, "/ibvs/error", 10
        )

        # ---------------- State ----------------------
        self.depth_img = None
        self.last_time = self.get_clock().now()

        self.desired_pts = self.desired_corners(
            Z_DES, FX, FY, CX, CY, TAG_SIZE
        )

        # Dynamic extension
        self.p_hat = np.zeros(3)      # virtual IBVS position
        self.p_pnp = np.zeros(3)      # anchor (PnP)

        # Tag visibility
        self.last_tag_time = None
        self.tag_lost = True
        self.TAG_TIMEOUT = 0.5
        self.DECAY_RATE = 0.9

        self.create_timer(0.1, self.timer_check_tag)
        self.get_logger().info("IBVS POSITION-BASED CONTROLLER ACTIVE")

    # ---------------------------------------------------------

    def desired_corners(self, Z, fx, fy, cx, cy, tag_size):
        half = tag_size / 2.0
        corners = np.array([
            [-half, -half, Z],
            [ half, -half, Z],
            [ half,  half, Z],
            [-half,  half, Z],
        ])

        pts = np.zeros((4, 2))
        for i, (X, Y, Z) in enumerate(corners):
            pts[i, 0] = fx * (X / Z) + cx
            pts[i, 1] = fy * (Y / Z) + cy
        return pts

    # ---------------------------------------------------------

    def cb_depth(self, msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001

    def cb_pnp(self, msg):
        self.p_pnp[0] = msg.x
        self.p_pnp[1] = msg.y
        self.p_pnp[2] = msg.z

    # ---------------------------------------------------------

    def interaction_matrix(self, u, v, Z):
        x = (u - CX) / FX
        y = (v - CY) / FY
        return np.array([
            [-1/Z, 0.0,  x/Z,  x*y, -(1+x*x),  y],
            [0.0, -1/Z,  y/Z,  1+y*y, -x*y,   -x]
        ])

    # ---------------------------------------------------------

    def cb_corners(self, msg):
        if self.depth_img is None:
            return

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        if dt <= 0:
            return

        self.last_tag_time = rclpy.time.Time.from_msg(msg.header.stamp)
        self.tag_lost = False

        rows = []
        errs = []

        pts = np.array([[p.x, p.y] for p in msg.polygon.points])
        h, w = self.depth_img.shape

        for i, (u, v) in enumerate(pts):
            ui, vi = int(u), int(v)
            if not (0 <= ui < w and 0 <= vi < h):
                return

            patch = self.depth_img[
                max(0, vi-PATCH):min(h, vi+PATCH+1),
                max(0, ui-PATCH):min(w, ui+PATCH+1)
            ]
            valid = patch[patch > 0]
            if valid.size == 0:
                return

            Z = float(np.median(valid))
            if Z < 0.2:
                return

            rows.append(self.interaction_matrix(u, v, Z))

            x, y = (u - CX)/FX, (v - CY)/FY
            xd, yd = (self.desired_pts[i] - [CX, CY]) / [FX, FY]
            errs.extend([x - xd, y - yd])

        # IBVS law
        L = np.vstack(rows)
        e = np.array(errs).reshape(-1, 1)

        mu = 1.0
        Vc = -LAMBDA_P * np.linalg.inv(
            L.T @ L + mu * np.eye(6)
        ) @ L.T @ e

        v_c = Vc[0:3]
        w_c = Vc[3:6]

        # Camera → body
        Wb = R_CB @ w_c
        Vb = (R_CB @ v_c) + np.cross(Wb.flatten(), P_CB).reshape(3, 1)

        # -------- Dynamic extension (POSITION based) --------
        v_ibvs = Vb.flatten()
        p_err = self.p_hat - self.p_pnp

        self.p_hat += (v_ibvs - K_ANCHOR * p_err) * dt
        self.p_hat = np.clip(self.p_hat, -MAX_OFFSET, MAX_OFFSET)

        # -------- Publish EXTERNAL POSITION to ArduSub --------
        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = float(self.p_hat[0])
        odom.pose.pose.position.y = float(self.p_hat[1])
        odom.pose.pose.position.z = float(self.p_hat[2])

        odom.pose.pose.orientation.w = 1.0

        self.odom_pub.publish(odom)

        # -------- Error logging --------
        err_msg = Float32MultiArray()
        err_msg.data = np.array(errs, dtype=np.float32).tolist()
        self.err_pub.publish(err_msg)

    # ---------------------------------------------------------

    def timer_check_tag(self):
        if self.last_tag_time is None:
            return

        now = self.get_clock().now()
        lost_dt = (now - self.last_tag_time).nanoseconds * 1e-9

        if lost_dt > self.TAG_TIMEOUT and not self.tag_lost:
            self.tag_lost = True
            self.get_logger().warn("AprilTag LOST → freezing external position")

            self.p_hat *= self.DECAY_RATE


def main():
    rclpy.init()
    node = IBVSControllerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
