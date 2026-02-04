#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import cv2
import math
from cv_bridge import CvBridge

from geometry_msgs.msg import PoseStamped, Point, PolygonStamped
from std_msgs.msg import Header

from ibvs.constants import *


def quaternion_from_yaw(yaw):
    """Yaw-only quaternion (roll = pitch = 0)"""
    return (
        0.0,
        0.0,
        math.sin(yaw * 0.5),
        math.cos(yaw * 0.5)
    )


class PNP_Node(Node):
    def __init__(self):
        super().__init__("PNP_Node")

        # Subscriber
        self.sub = self.create_subscription(PolygonStamped, "/apriltag/corners", self.cb, 10)

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, "/mavros/vision_pose/pose", 10)
        self.rel_pub = self.create_publisher(Point, "/pnp/relative_position", 10)

        # Camera intrinsics
        self.camera_matrix = np.array([
            [FX,  0, CX],
            [0, FY, CY],
            [0,  0,  1]
        ], dtype=np.float32)

        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        # AprilTag object points (tag frame)
        s = TAG_SIZE * 0.5
        self.object_pts = np.array([
            [-s, -s, 0],  # matches c[3]
            [ s, -s, 0],  # matches c[2]
            [ s,  s, 0],  # matches c[1]
            [-s,  s, 0],  # matches c[0]
        ], dtype=np.float32)

        self.world_locked = False
        self.R_wt = None
        self.t_wt = None
        
        self.get_logger().info("PnP Node Initialized")

    # ---------------------------------------------------------

    def cb(self, msg: PolygonStamped):
        if len(msg.polygon.points) != 4:
            return

        img_pts = np.array(
            [[p.x, p.y] for p in msg.polygon.points], dtype=np.float32)

        ok, rvec, tvec = cv2.solvePnP(
            self.object_pts, img_pts,
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)

        if not ok or tvec[2] <= 0.0:
            return

        R_ct, _ = cv2.Rodrigues(rvec)
        t_ct = tvec.reshape(3, 1)

        # Lock world frame on first detection
        if not self.world_locked:
            self.R_wt = np.eye(3)
            self.t_wt = np.zeros((3, 1))
            self.world_locked = True
            self.get_logger().info("World frame locked to AprilTag")

        # Camera pose in world (tag) frame
        R_wc = self.R_wt @ R_ct.T
        t_wc = self.R_wt @ (-R_ct.T @ t_ct) + self.t_wt

        # Optical → ENU-like proxy (consistent with IBVS)
        x = float(t_wc[2])
        y = float(-t_wc[0])
        z = float(-t_wc[1])

        # Yaw from world-aligned rotation
        yaw = math.atan2(R_wc[1, 0], R_wc[0, 0])
        qx, qy, qz, qw = yaw_to_quaternion(yaw)

        # Debug / IBVS-relative position
        self.rel_pub.publish(Point(x=x, y=y, z=z))

        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = "vision"

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z

        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        self.pose_pub.publish(pose)


def main():
    rclpy.init()
    rclpy.spin(PNP_Node())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
