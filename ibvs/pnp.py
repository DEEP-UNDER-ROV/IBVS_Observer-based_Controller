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
            [-s,  s, 0],
            [ s,  s, 0],
            [ s, -s, 0],
            [-s, -s, 0]
        ], dtype=np.float32)

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

        # Rotation: tag -> camera
        R_cm, _ = cv2.Rodrigues(rvec)
        t_cm = tvec.reshape(3, 1)

        # Invert transform: camera pose in tag(world) frame
        R_mc = R_cm.T
        t_mc = -R_cm.T @ t_cm

        # Camera optical → ROS ENU (or body-aligned proxy)
        x = float(t_mc[2])      # forward
        y = float(-t_mc[0])     # right
        z = float(-t_mc[1])     # down

        # Yaw from rotation
        yaw = math.atan2(R_mc[1, 0], R_mc[0, 0])
        qx, qy, qz, qw = quaternion_from_yaw(yaw)

        # Publish relative position (debug / IBVS)
        self.rel_pub.publish(Point(x=x, y=y, z=z))

        # Publish MAVROS vision pose
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
