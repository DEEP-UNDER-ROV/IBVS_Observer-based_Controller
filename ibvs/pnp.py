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
            [ s,  s, 0],  # matches c[3]
            [-s,  s, 0],  # matches c[2]
            [-s, -s, 0],  # matches c[1]
            [ s, -s, 0],  # matches c[0]
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

        if not ok:
            return

        t_ct = tvec.reshape(3)

        # ---------- CRITICAL VALIDITY CHECK ----------
        # Tag must be in front of camera
        if t_ct[2] <= 0.1:
            return

        R_ct, _ = cv2.Rodrigues(rvec)

        # ---------- WORLD LOCK ----------
        if not self.world_locked:
            self.R_wt = R_ct.T
            self.t_wt = -R_ct.T @ t_ct.reshape(3, 1)
            self.world_locked = True
            self.get_logger().info("World frame locked to AprilTag")

            lock_msg = Bool()
            lock_msg.data = True
            self.pub_lock.publish(lock_msg)

        # ---------- EXPRESS TAG IN WORLD FRAME ----------
        t_wt = self.R_wt @ t_ct.reshape(3, 1) + self.t_wt

        # ---------- PUBLISH ----------
        pose = PoseStamped()
        pose.header = msg.header
        pose.header.frame_id = "world"

        pose.pose.position.x = float(t_wt[0])
        pose.pose.position.y = float(t_wt[1])
        pose.pose.position.z = float(t_wt[2])

        # Orientation (camera-relative, optional)
        q = self.rotation_to_quaternion(self.R_wt @ R_ct)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        self.pub_pose.publish(pose)

    # ----------------------------------------------------

    @staticmethod
    def rotation_to_quaternion(R):
        qw = np.sqrt(1.0 + np.trace(R)) / 2.0
        qx = (R[2, 1] - R[1, 2]) / (4.0 * qw)
        qy = (R[0, 2] - R[2, 0]) / (4.0 * qw)
        qz = (R[1, 0] - R[0, 1]) / (4.0 * qw)
        return qx, qy, qz, qw


def main():
    rclpy.init()
    rclpy.spin(PNP_Node())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
