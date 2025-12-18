#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import math
from geometry_msgs.msg import PolygonStamped, PoseStamped

from ibvs_control.constants import FX, FY, CX, CY, TAG_SIZE, LAMBDA_P, PATCH, Z_DES, P_CB_X, P_CB_Y, P_CB_Z, MAX_LIN_VEL, MAX_ANG_VEL, DIST_COEFFS


class PnPPoseNode(Node):
    def __init__(self):
        super().__init__("pnp_pose_node")

        # Subscribers
        self.sub = self.create_subscription(
            PolygonStamped, "/apriltag/corners", self.cb, 10
        )

        # Publishers - Sending to MAVROS for EKF2 fusion
        self.pose_pub = self.create_publisher(
            PoseStamped, "/mavros/vision_pose/pose", 10
        )

        # Camera calibration matrices
        self.camera_matrix = np.array([
            [FX, 0, CX],
            [0, FY, CY],
            [0,  0,  1]
        ], dtype=np.float32)

        # FIXED: Assigning the calibrated coefficients from constants.py
        # OpenCV solvePnP expects [k1, k2, p1, p2, k3]
        self.dist_coeffs = np.array(DIST_COEFFS, dtype=np.float32)

        # Define 3D points of the tag in the Tag Frame (Flat on Z=0)
        # Order must match the CCW ordering from your detector
        s = TAG_SIZE / 2.0
        self.object_points = np.array([
            [-s, -s, 0], # Top Left
            [ s, -s, 0], # Top Right
            [ s,  s, 0], # Bottom Right
            [-s,  s, 0]  # Bottom Left
        ], dtype=np.float32)

        self.get_logger().info("PnP Pose Node initialized with hardcoded intrinsics")

    def cb(self, msg):
        if len(msg.polygon.points) != 4:
            return

        # 1. Extract 2D image points from the message
        image_pts = np.array(
            [[p.x, p.y] for p in msg.polygon.points],
            dtype=np.float32
        )

        # 2. Solve PnP (Using IPPE_SQUARE for planar AprilTags is highly accurate)
        success, rvec, tvec = cv2.solvePnP(
            self.object_points,
            image_pts,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if not success:
            return

        # 3. Coordinate Transformation
        # solvePnP tvec is [x_cam, y_cam, z_cam] 
        # We need to map this to MAVROS [x_map, y_map, z_map]
        # Common mapping for a front-facing camera:
        # Map X = Cam Z (Depth)
        # Map Y = -Cam X (Left)
        # Map Z = -Cam Y (Up)
        
        tx, ty, tz = tvec.flatten()

        pose = PoseStamped()
        pose.header = msg.header
        pose.header.frame_id = "odom" # Or "map" depending on your EKF setup

        # Position mapping
        pose.pose.position.x = float(tz)  # Forward
        pose.pose.position.y = float(-tx) # Left
        pose.pose.position.z = float(-ty) # Up

        # 4. Rotation/Orientation logic
        R, _ = cv2.Rodrigues(rvec)
        
        # Convert Rotation Matrix to Euler (Roll, Pitch, Yaw)
        # We need to account for the frame swap here too
        yaw = math.atan2(R[1, 0], R[0, 0])
        pitch = math.asin(-R[2, 0])
        roll = math.atan2(R[2, 1], R[2, 2])

        q = self.euler_to_quaternion(roll, pitch, yaw)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

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
    node = PnPPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
