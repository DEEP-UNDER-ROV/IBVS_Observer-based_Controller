#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import math

from geometry_msgs.msg import PolygonStamped, PoseStamped

from ibvs_control.constants import FX, FY, CX, CY, TAG_SIZE, DIST_COEFFS


class PnPRelativePoseNode(Node):

    def rotation_matrix_to_euler(self, R):
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.asin(-R[2, 0])
        yaw = math.atan2(R[1, 0], R[0, 0])
        return roll, pitch, yaw


    def __init__(self):
        super().__init__("pnp_relative_pose_node")

        self.sub = self.create_subscription(
            PolygonStamped, "/apriltag/corners", self.cb, 10
        )

        self.pub = self.create_publisher(
            PoseStamped, "/mavros/vision_pose/pose", 10
        )

        self.camera_matrix = np.array([
            [FX, 0, CX],
            [0, FY, CY],
            [0,  0,  1]
        ], dtype=np.float32)

        self.dist_coeffs = np.array(DIST_COEFFS, dtype=np.float32)

        s = TAG_SIZE / 2.0
        self.object_points = np.array([
            [-s, -s, 0],
            [ s, -s, 0],
            [ s,  s, 0],
            [-s,  s, 0]
        ], dtype=np.float32)

        # Local reference
        self.origin_set = False
        self.t0 = None
        self.R0 = None

        self.get_logger().info("PnP RELATIVE pose node initialized")

    def cb(self, msg):
        if len(msg.polygon.points) != 4:
            return

        image_pts = np.array(
            [[p.x, p.y] for p in msg.polygon.points],
            dtype=np.float32
        )

        success, rvec, tvec = cv2.solvePnP(
            self.object_points,
            image_pts,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if not success:
            return

        # Reject flipped IPPE solution
        if tvec[2] <= 0:
            return

        R, _ = cv2.Rodrigues(rvec)

        # ===== DEBUG: RAW PnP OUTPUT =====
        raw_roll, raw_pitch, raw_yaw = self.rotation_matrix_to_euler(R)

        self.get_logger().info(
            "[RAW PnP] tvec (cam frame) [m]: "
            f"x={tvec[0][0]:+.3f}, y={tvec[1][0]:+.3f}, z={tvec[2][0]:+.3f} | "
            "RPY [deg]: "
            f"roll={math.degrees(raw_roll):+.1f}, "
            f"pitch={math.degrees(raw_pitch):+.1f}, "
            f"yaw={math.degrees(raw_yaw):+.1f}"
        )

        # Set local origin on first detection
        if not self.origin_set:
            self.t0 = tvec.copy()
            self.R0 = R.copy()
            self.origin_set = True
            self.get_logger().info("Vision local origin initialized")
            return

        # Relative translation (tag frame)
        t_rel = tvec - self.t0

        # Relative rotation
        R_rel = self.R0.T @ R

        # ===== DEBUG: RELATIVE POSE =====
        rel_roll, rel_pitch, rel_yaw = self.rotation_matrix_to_euler(R_rel)

        self.get_logger().info(
            "[RELATIVE] position [m]: "
            f"x={t_rel[2][0]:+.4f}, "
            f"y={-t_rel[0][0]:+.4f}, "
            f"z={-t_rel[1][0]:+.4f} | "
            "RPY [deg]: "
            f"roll={math.degrees(rel_roll):+.2f}, "
            f"pitch={math.degrees(rel_pitch):+.2f}, "
            f"yaw={math.degrees(rel_yaw):+.2f}"
        )

        # Camera frame → body frame (ENU)
        tx, ty, tz = t_rel.flatten()

        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = "vision"

        pose.pose.position.x = float(tz)
        pose.pose.position.y = float(-tx)
        pose.pose.position.z = float(-ty)

        # Rotation → quaternion
        yaw = math.atan2(R_rel[1, 0], R_rel[0, 0])
        pitch = math.asin(-R_rel[2, 0])
        roll = math.atan2(R_rel[2, 1], R_rel[2, 2])

        q = self.euler_to_quaternion(roll, pitch, yaw)

        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        self.pub.publish(pose)

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
    node = PnPRelativePoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

