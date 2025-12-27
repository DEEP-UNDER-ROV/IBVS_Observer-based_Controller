#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import cv2
import math

from geometry_msgs.msg import PoseStamped, Point, PolygonStamped, Point32
from cv_bridge import CvBridge
from std_msgs.msg import Header

from ibvs.constants import *


def quaternion_from_yaw(yaw):
    """
    Convert yaw angle (rad) to quaternion.
    Roll = pitch = 0 (valid for ArduSub EKF yaw fusion)
    """
    half = 0.5 * yaw
    return (
        0.0,                  # qx
        0.0,                  # qy
        math.sin(half),       # qz
        math.cos(half)        # qw
    )

class PNP_Node(Node):
    def __init__(self):
        super().__init__("PNP_Node")
        self.bridge = CvBridge()

        #Subscriber
        self.sub = self.create_subscription(PolygonStamped, "/apriltag/corners", self.cb, 10)

        #Publisher
        self.pose_pub = self.create_publisher(PoseStamped, "/mavros/vision_pose/pose", 10)
        self.rel_pub = self.create_publisher(Point, "/pnp/relative_position", 10)

        self.camera_matrix = np.array([
            [FX, 0, CX],
            [0, FY, CY],
            [0,  0,  1]
        ], dtype=np.float32)

        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        s = TAG_SIZE / 2.0
        self.object_pts = np.array([
            [-s,  s, 0],
            [ s,  s, 0],
            [ s, -s, 0],
            [-s, -s, 0]
        ], dtype=np.float32)

        self.origin_set = False
        self.t0 = None
        self.R0 = None

        self.get_logger().info("PnP node initialized")

    # ---------------------------------------------------------

    def cb(self, msg):
        if len(msg.polygon.points) != 4:
            return

        img_pts = np.array(
            [[p.x, p.y] for p in msg.polygon.points], dtype=np.float32)

        ok, rvec, tvec = cv2.solvePnP(
            self.object_pts, img_pts,
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)

        if not ok or tvec[2] <= 0:
            return

        R, _ = cv2.Rodrigues(rvec)

        if not self.origin_set:
            self.t0 = tvec.copy()
            self.R0 = R.copy()
            self.origin_set = True
            self.get_logger().info("PnP reference initialized")
            return

        # Relative pose (tag frame)
        t_rel = self.R0.T @ (tvec - self.t0)

        # Relative rotation
        R_rel = self.R0.T @ R

        x = float(t_rel[2])     # forward → X
        y = float(-t_rel[0])    # right → -Y
        z = float(-t_rel[1])    # down → -Z

        yaw = math.atan2(R_rel[1, 0], R_rel[0, 0])
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        self.rel_pub.publish(Point(x=x, y=y, z=z))

        pose = PoseStamped()
        pose.header = msg.header
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        self.pose_pub.publish(pose))


    @staticmethod
    def euler_to_quaternion(roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        return (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy
        )

def main():
    rclpy.init()
    rclpy.spin(PNP_Node())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
