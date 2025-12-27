# New PNP For Position Only

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

        self.dist_coeffs = np.zeros((5, 1))

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

        # Publish relative translation for IBVS anchor
        self.rel_pub.publish(Point(
            x=float(t_rel[2]),
            y=float(-t_rel[0]),
            z=float(-t_rel[1])
        ))

        # Publish to MAVROS EKF
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose.position.x = float(t_rel[2])
        pose.pose.position.y = float(-t_rel[0])
        pose.pose.position.z = float(-t_rel[1])
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)


def main():
    rclpy.init()
    rclpy.spin(PNP_Node())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
