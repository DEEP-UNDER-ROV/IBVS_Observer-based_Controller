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


class IBVSVelocityController(Node):
    def __init__(self):
        super().__init__("IBVSControllerNode_Velocity")

        # Subscribers
        self.sub_corners = self.create_subscription(PolygonStamped, "/apriltag/corners", self.cb_corners, 10)

        # Publishers
        self.vel_pub = self.create_publisher(Twist, "/mavros/setpoint_velocity/cmd_vel_unstamped", 10)
        self.err_pub = self.create_publisher(Float32MultiArray, "/ibvs/error", 10)
      
        self.desired_pts = self.compute_desired_corners( Z_DES, FX, FY, CX, CY, TAG_SIZE)
        self.get_logger().info("IBVS VELOCITY controller running")

    # -------------------------------------------------

    @staticmethod
    def compute_desired_corners(Z, fx, fy, cx, cy, tag_size):
        h = tag_size / 2.0
        obj = np.array([
            [-h, -h, Z],
            [ h, -h, Z],
            [ h,  h, Z],
            [-h,  h, Z],
        ])

        img = np.zeros((4, 2))
        for i, (X, Y, Zc) in enumerate(obj):
            img[i, 0] = fx * X / Zc + cx
            img[i, 1] = fy * Y / Zc + cy
        return img

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
        if len(msg.polygon.points) != 4:
            return

        rows = []
        errs = []

        for i, p in enumerate(msg.polygon.points):
            u, v, Z = p.x, p.y, p.z
            if Z <= 0.2:
                return

            rows.append(self.interaction_matrix(u, v, Z))

            x, y = (u - CX)/FX, (v - CY)/FY
            xd, yd = (self.desired_pts[i] - [CX, CY]) / [FX, FY]
            errs.extend([x - xd, y - yd, Z - Z_DES])

        L = np.vstack(rows)
        e = np.array(errs).reshape(-1, 1)

        mu = 0.01
        Vc = -LAMBDA_P * np.linalg.inv(L.T @ L + mu * np.eye(6)) @ L.T @ e

        v_c = Vc[0:3]
        w_c = Vc[3:6]

        # Camera → body
        Wb = R_CB @ w_c
        Vb = (R_CB @ v_c) + np.cross(Wb.flatten(), P_CB).reshape(3, 1)

        cmd = Twist()
        cmd.linear.x = float(np.clip(Vb[0], -MAX_LIN_VEL, MAX_LIN_VEL))
        cmd.linear.y = float(np.clip(Vb[1], -MAX_LIN_VEL, MAX_LIN_VEL))
        cmd.linear.z = float(np.clip(Vb[2], -MAX_LIN_VEL, MAX_LIN_VEL))
        cmd.angular.z = float(np.clip(Wb[2], -MAX_ANG_VEL, MAX_ANG_VEL))

        self.vel_pub.publish(cmd)

        err_msg = Float32MultiArray()
        err_msg.data = errs
        self.err_pub.publish(err_msg)


def main():
    rclpy.init()
    rclpy.spin(IBVSVelocityController())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
