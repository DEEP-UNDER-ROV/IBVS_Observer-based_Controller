# IBVS Integral Velocity -> Full LS with Yaw

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math

from geometry_msgs.msg import PolygonStamped, Twist, Point, PoseStamped
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ibvs.constants import *

class IBVSControllerNode(Node):
    def __init__(self):
        super().__init__("IBVSControllerNode")
        self.bridge = CvBridge()

        # Subscriber
        self.sub_corners = self.create_subscription(PolygonStamped, "/apriltag/corners", self.cb_corners, 10)
        self.sub_depth = self.create_subscription(Image, "/camera/depth/image_raw", self.cb_depth, 10)
        self.sub_pnp = self.create_subscription(Point, "/pnp/relative_position", self.cb_pnp, 10)
#        self.sub_pnp_pose = self.create_subscription(PoseStamped, "/mavros/vision_pose/pose", self.cb_pnp_pose, 10)
        self.sub_ekf = self.create_subscription(PoseStamped, "/mavros/local_position/pose", self.cb_ekf, 10)

        # Publisher
        self.target_pub = self.create_publisher(PoseStamped, "/mavros/setpoint_position/local", 10)
        self.vel_pub = self.create_publisher(Twist, "/ibvs/vel", 10)
        self.pos_pub = self.create_publisher(Point, "/ibvs/pos", 10)
        self.err_pub = self.create_publisher(Float32MultiArray, "/ibvs/error", 10)

        self.depth_img = None
        self.current_pose = None
        self.last_time = self.get_clock().now()
        # Desired image features
        self.desired_pts = self.desired_corners(Z_DES, FX, FY, CX, CY, TAG_SIZE)

        # >>> ADDED: PnP + IBVS position correction <<<
        self.p_pnp = np.zeros(3)     # tag-relative position
        self.p_corr = np.zeros(3)    # IBVS correction integral
        self.p_cmd = np.zeros(3)
        self.yaw = 0.0   # fused yaw (rad)

        # >>> ADDED: Tag-loss watchdog <<<
        self.last_tag_time = None
        self.tag_lost = True
        self.TAG_TIMEOUT = 0.5  # seconds

        self.get_logger().info("CAUTION !! IBVS Control ON")

        # Timer for tag-loss handling
        self.create_timer(0.1, self.timer_check_tag)

    
    def desired_corners(self, Z_DES, fx, fy, cx, cy, tag_size):
        half = tag_size / 2.0
        corners = np.array([
            [-half, -half, Z_DES],
            [ half, -half, Z_DES],
            [ half,  half, Z_DES],
            [-half,  half, Z_DES],
        ])

        pts = np.zeros((4, 2), dtype=float)
        for i, (X, Y, Z) in enumerate(corners):
            pts[i, 0] = fx * X / Z + cx
            pts[i, 1] = fy * Y / Z + cy
        return pts
    @staticmethod
    def quaternion_from_yaw(yaw):
        half = 0.5 * yaw
        return (0.0, 0.0, math.sin(half), math.cos(half))
        
    def cb_depth(self, msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001


    def cb_pnp(self, msg):
        self.p_pnp[0] = msg.x
        self.p_pnp[1] = msg.y
        self.p_pnp[2] = msg.z

    # def cb_pnp_pose(self, msg):
    #     q = msg.pose.orientation

    
    def cb_ekf(self, msg):
        self.current_pose = msg
        q = msg.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z), 1.0 - 2.0 * (q.z * q.z))

    def interaction_matrix(self, u, v, Z):
        x = (u - CX) / FX
        y = (v - CY) / FY
        return np.array([
            [-1/Z,  0,    x/Z,  x*y,      -(1+x*x),  y],
            [0,    -1/Z,  y/Z,  1+y*y,    -x*y,     -x],
            [0,     0,   -1,   -y*Z,       x*Z,      0]
        ])


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

        rows, errs = [], []
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
            errs.extend([x - xd, y - yd, Z - Z_DES])

        L = np.vstack(rows)
        e = np.array(errs).reshape(-1, 1)

        mu = 0.01
        Vc = -LAMBDA_P * np.linalg.inv(L.T @ L + mu * np.eye(6)) @ L.T @ e

        v_c = Vc[0:3].reshape(3, 1)
        w_c = Vc[3:6].reshape(3, 1)

        Wb = R_CB @ w_c
        Vb = (R_CB @ v_c) + np.cross(Wb.flatten(), P_CB).reshape(3, 1)

        vel = Twist()
        vel.linear.x = float(np.clip(Vb[0], -MAX_LIN_VEL, MAX_LIN_VEL))
        vel.linear.y = float(np.clip(Vb[1], -MAX_LIN_VEL, MAX_LIN_VEL))
        vel.linear.z = float(np.clip(Vb[2], -MAX_LIN_VEL, MAX_LIN_VEL))
        vel.angular.z = float(np.clip(Wb[2], -MAX_ANG_VEL, MAX_ANG_VEL))
        self.vel_pub.publish(vel)

                             
        pos = self.current_pose.pose.position
        setpoint = PoseStamped()
        setpoint.header.stamp = now.to_msg()
        setpoint.header.frame_id = "map"
        setpoint.pose.position.x = pos.x + vel.linear.x * dt
        setpoint.pose.position.y = pos.y + vel.linear.y * dt
        setpoint.pose.position.z = pos.z + vel.linear.z * dt
        setpoint.pose.orientation = self.current_pose.pose.orientation

        self.target_pub.publish(setpoint)

        # ---------------- POSITION CONTROL (REMODELED) ----------------
        # IBVS correction only
        # self.p_corr += Vb.flatten() * dt
        # self.p_corr = np.clip(self.p_corr, -MAX_OFFSET, MAX_OFFSET)

        # # >>> FINAL POSITION COMMAND <<<
        # self.p_cmd = self.p_pnp + self.p_corr

        # qx, qy, qz, qw = quaternion_from_yaw(self.yaw)

        # odom = Odometry()
        # odom.header.stamp = now.to_msg()
        # odom.header.frame_id = "map"
        # odom.child_frame_id = "base_link"
        
        # odom.pose.pose.position.x = float(self.p_cmd[0])
        # odom.pose.pose.position.y = float(self.p_cmd[1])
        # odom.pose.pose.position.z = float(self.p_cmd[2])
        
        # odom.pose.pose.orientation.x = qx
        # odom.pose.pose.orientation.y = qy
        # odom.pose.pose.orientation.z = qz
        # odom.pose.pose.orientation.w = qw
        
        # self.odom_pub.publish(odom)

        err_msg = Float32MultiArray()
        err_msg.data = np.array(errs, dtype=np.float32).tolist()
        self.err_pub.publish(err_msg)

    # ---------------------------------------------------------
    # >>> TAG LOSS HANDLER <<<
    def timer_check_tag(self):
        if self.last_tag_time is None:
            return

        now = self.get_clock().now()
        dt_lost = (now - self.last_tag_time).nanoseconds * 1e-9

        if dt_lost > self.TAG_TIMEOUT and not self.tag_lost:
            self.tag_lost = True
            self.get_logger().warn("AprilTag LOST → freezing external position")
            self.vel_pub.publish(Twist())

            if self.current_pose:
                stop_point = Point()
                stop_point.x = self.current_pose.pose.position.x
                stop_point.y = self.current_pose.pose.position.y
                stop_point.z = self.current_pose.pose.position.z
                self.pos_pub.publish(stop_point)
            ))

def main():
    rclpy.init()
    rclpy.spin(IBVSControllerNode())
    rclpy.shutdown()

if __name__ == "__main__":
    main()
