#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import PolygonStamped, Twist, Point
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ibvs.constants import *


class IBVSControllerNode(Node):
    def __init__(self):
        super().__init__("IBVSControllerNode")
        self.bridge = CvBridge()

        # Subscribers
        self.sub_corners = self.create_subscription(
            PolygonStamped, "/apriltag/corners", self.cb_corners, 10)
        self.sub_depth = self.create_subscription(
            Image, "/camera/depth/image_raw", self.cb_depth, 10)
        self.sub_pnp = self.create_subscription(
            Point, "/pnp/relative_position", self.cb_pnp, 10)

        # Publishers
        self.vel_pub = self.create_publisher(Twist, "/ibvs/vel", 10)
        self.pos_pub = self.create_publisher(Point, "/ibvs/pos", 10)
        self.err_pub = self.create_publisher(Float32MultiArray, "/ibvs/error", 10)

        self.depth_img = None
        self.last_time = self.get_clock().now()

        # Desired image features
        self.desired_pts = self.desired_corners(
            Z_DES, FX, FY, CX, CY, TAG_SIZE)

        # --- Dynamic extension state ---
        self.p_hat = np.zeros(3)       # virtual IBVS position
        self.p_pnp = np.zeros(3)       # anchor (from PnP)

        self.get_logger().info("IBVS Controller (pure x,y + anchored integrator) started")

    # ---------------------------------------------------------

    def desired_corners(self, Z, fx, fy, cx, cy, tag_size):
        s = tag_size / 2.0
        corners = np.array([
            [-s, -s, Z],
            [ s, -s, Z],
            [ s,  s, Z],
            [-s,  s, Z],
        ])

        pts = np.zeros((4, 2))
        for i, (X, Y, Z) in enumerate(corners):
            pts[i, 0] = fx * X / Z + cx
            pts[i, 1] = fy * Y / Z + cy
        return pts

    # ---------------------------------------------------------

    def cb_depth(self, msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001

    def cb_pnp(self, msg):
        # Relative pose anchor (tag frame)
        self.p_pnp[:] = np.array([msg.x, msg.y, msg.z])

    # ---------------------------------------------------------

    def interaction_matrix(self, x, y, Z):
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

            # normalized coordinates
            x = (u - CX) / FX
            y = (v - CY) / FY
            xd = (self.desired_pts[i, 0] - CX) / FX
            yd = (self.desired_pts[i, 1] - CY) / FY

            rows.append(self.interaction_matrix(x, y, Z))
            errs.extend([x - xd, y - yd])

        # --- IBVS control law ---
        L = np.vstack(rows)                  # 8x6
        e = np.array(errs).reshape(-1, 1)    # 8x1

        mu = 0.01
        Vc = -LAMBDA_P * np.linalg.inv(
            L.T @ L + mu * np.eye(6)) @ L.T @ e

        v_c = Vc[0:3]
        w_c = Vc[3:6]

        # Camera → body
        Wb = R_CB @ w_c
        Vb = (R_CB @ v_c) + np.cross(Wb.flatten(), P_CB).reshape(3, 1)

        # --- Dynamic extension (anchored integrator) ---
        v_ibvs = Vb.flatten()
        p_err = self.p_hat - self.p_pnp
        self.p_hat += (v_ibvs - K_ANCHOR * p_err) * dt
        self.p_hat = np.clip(self.p_hat, -MAX_OFFSET, MAX_OFFSET)

        # Publish position command
        self.pos_pub.publish(Point(
            x=float(self.p_hat[0]),
            y=float(self.p_hat[1]),
            z=float(self.p_hat[2])
        ))

        # (Optional) publish velocity for logging
        vel = Twist()
        vel.linear.x = float(v_ibvs[0])
        vel.linear.y = float(v_ibvs[1])
        vel.linear.z = float(v_ibvs[2])
        vel.angular.z = float(Wb[2])
        self.vel_pub.publish(vel)

        # Error logging
        err_msg = Float32MultiArray()
        err_msg.data = errs
        self.err_pub.publish(err_msg)


def main():
    rclpy.init()
    rclpy.spin(IBVSControllerNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
