#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import math

from geometry_msgs.msg import PolygonStamped, Twist, PoseStamped
from std_msgs.msg import Float32MultiArray

from ibvs.constants import *


class IBVSControllerNode(Node):
    """
    Pure IBVS controller:
    - Input: corners (u, v, Z) from detector
    - Output: body velocity command
    """

    def __init__(self):
        super().__init__("ibvs_controller")

        # ---------------- Subscribers ----------------
        self.sub_corners = self.create_subscription(
            PolygonStamped,
            "/apriltag/corners_depth",
            self.cb_corners,
            10
        )

        self.sub_pose = self.create_subscription(
            PoseStamped,
            "/mavros/local_position/pose",
            self.cb_pose,
            10
        )

        # ---------------- Publishers ----------------
        self.vel_pub = self.create_publisher(Twist, "/ibvs/vel", 10)
        self.err_pub = self.create_publisher(Float32MultiArray, "/ibvs/error", 10)

        # ---------------- State ----------------
        self.current_pose = None
        self.last_time = self.get_clock().now()

        # Desired image features (pixels)
        self.desired_pts = self.desired_corners(
            Z_DES, FX, FY, CX, CY, TAG_SIZE
        )

        self.get_logger().info("IBVS Controller (depth-optimized) started")

    # -------------------------------------------------

    def cb_pose(self, msg):
        self.current_pose = msg

    # -------------------------------------------------

    @staticmethod
    def desired_corners(Z, fx, fy, cx, cy, tag_size):
        half = tag_size / 2.0
        corners = np.array([
            [-half, -half, Z],
            [ half, -half, Z],
            [ half,  half, Z],
            [-half,  half, Z],
        ])

        pts = np.zeros((4, 2))
        for i, (X, Y, Zc) in enumerate(corners):
            pts[i, 0] = fx * X / Zc + cx
            pts[i, 1] = fy * Y / Zc + cy
        return pts

    # -------------------------------------------------

    def interaction_matrix(self, u, v, Z):
        x = (u - CX) / FX
        y = (v - CY) / FY

        return np.array([
            [-1/Z,  0,    x/Z,   x*y,     -(1 + x*x),  y],
            [0,    -1/Z,  y/Z,   1 + y*y, -x*y,       -x],
            [0,     0,   -1,    -y*Z,      x*Z,        0]
        ])

    # -------------------------------------------------

    def cb_corners(self, msg):
        if len(msg.polygon.points) != 4 or self.current_pose is None:
            return

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        if dt <= 0.0:
            return

        rows = []
        errs = []

        for i, p in enumerate(msg.polygon.points):
            u, v, Z = p.x, p.y, p.z
            if Z <= 0.1:
                return

            rows.append(self.interaction_matrix(u, v, Z))

            x, y = (u - CX) / FX, (v - CY) / FY
            xd, yd = (self.desired_pts[i] - [CX, CY]) / [FX, FY]

            errs.extend([x - xd, y - yd, Z - Z_DES])

        L = np.vstack(rows)
        e = np.array(errs).reshape(-1, 1)

        mu = 0.01
        Vc = -LAMBDA_P * np.linalg.inv(
            L.T @ L + mu * np.eye(6)
        ) @ L.T @ e

        v_c = Vc[0:3]
        w_c = Vc[3:6]

        # Camera → body transform
        Wb = R_CB @ w_c
        Vb = (R_CB @ v_c) + np.cross(Wb.flatten(), P_CB).reshape(3, 1)

        vel = Twist()
        vel.linear.x = float(np.clip(Vb[0], -MAX_LIN_VEL, MAX_LIN_VEL))
        vel.linear.y = float(np.clip(Vb[1], -MAX_LIN_VEL, MAX_LIN_VEL))
        vel.linear.z = float(np.clip(Vb[2], -MAX_LIN_VEL, MAX_LIN_VEL))
        vel.angular.z = float(np.clip(Wb[2], -MAX_ANG_VEL, MAX_ANG_VEL))

        self.vel_pub.publish(vel)

        err_msg = Float32MultiArray()
        err_msg.data = np.array(errs, dtype=np.float32).tolist()
        self.err_pub.publish(err_msg)


def main():
    rclpy.init()
    rclpy.spin(IBVSControllerNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
