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
    
        # Image points (must match object_pts order!)
        img_pts = np.array(
            [[p.x, p.y] for p in msg.polygon.points],
            dtype=np.float32
        )
    
        ok, rvec, tvec = cv2.solvePnP(
            self.object_pts,
            img_pts,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
    
        if not ok:
            return
    
        t_ct = tvec.reshape(3)
    
        # Must be in front of camera
        if t_ct[2] <= 0.05:
            return
    
        R_ct, _ = cv2.Rodrigues(rvec)
    
        # ---------- CRITICAL: PLANAR DISAMBIGUATION ----------
        # Tag Z-axis (normal) must point toward camera
        # Tag normal in camera frame = third column of R_ct
        if R_ct[2, 2] <= 0.0:
            return
    
        # ---------- CAMERA POSE IN WORLD (TAG) FRAME ----------
        # world == tag
        R_wc = R_ct.T
        t_wc = -R_ct.T @ t_ct.reshape(3, 1)
    
        # ---------- COORDINATE REMAP (camera optical → body proxy) ----------
        # Optical: x right, y down, z forward
        # Body-like: x forward, y right, z down
        x = float(t_wc[2])
        y = float(-t_wc[0])
        z = float(-t_wc[1])
    
        # ---------- YAW-ONLY ORIENTATION ----------
        yaw = math.atan2(R_wc[1, 0], R_wc[0, 0])
        qx = 0.0
        qy = 0.0
        qz = math.sin(0.5 * yaw)
        qw = math.cos(0.5 * yaw)
    
        # ---------- DEBUG / IBVS ----------
        self.rel_pub.publish(Point(x=x, y=y, z=z))
    
        # ---------- MAVROS VISION POSE ----------
        pose = PoseStamped()
        pose.header = msg.header
        pose.header.frame_id = "tag_world"
    
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
    
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
    
        self.pose_pub.publish(pose)

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
