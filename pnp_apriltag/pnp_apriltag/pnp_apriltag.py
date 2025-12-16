#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import cv2
import math
import pyrealsense2 as rs

from pupil_apriltags import Detector

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Range


class AprilTagPnPVision(Node):

    def __init__(self):
        super().__init__('apriltag_pnp_vision')

        # ===============================
        # Publishers
        # ===============================
        self.vision_pub = self.create_publisher(
            PoseStamped,
            '/mavros/vision_pose/pose',
            10
        )

        self.depth_pub = self.create_publisher(
            Range,
            '/mavros/distance_sensor/depth',
            10
        )

        # ===============================
        # RealSense pipeline (COLOR ONLY)
        # ===============================
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)

        # ===============================
        # Camera intrinsics
        # ===============================
        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()

        self.fx = intr.fx
        self.fy = intr.fy
        self.cx = intr.ppx
        self.cy = intr.ppy

        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        self.dist_coeffs = np.zeros((4, 1))

        # ===============================
        # AprilTag detector
        # ===============================
        self.detector = Detector(
            families="tag36h11",
            nthreads=2,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=True,
        )

        # ===============================
        # Tag geometry (meters)
        # ===============================
        self.tag_size = 0.16
        s = self.tag_size / 2.0

        self.object_points = np.array([
            [-s, -s, 0],
            [ s, -s, 0],
            [ s,  s, 0],
            [-s,  s, 0],
        ], dtype=np.float32)

        # ===============================
        # Timer (20 Hz)
        # ===============================
        self.timer = self.create_timer(0.05, self.loop)

        self.get_logger().info("AprilTag PnP Vision + Depth node started")

    # ============================================================
    def loop(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return

        color = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        detections = self.detector.detect(gray)

        if len(detections) == 0:
            self.show_view(color, None)
            return

        tag = detections[0]
        image_points = np.array(tag.corners, dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            self.object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if not success:
            self.show_view(color, None)
            return

        # ========================================================
        # Translation (OpenCV camera frame)
        # X right, Y down, Z forward
        # ========================================================
        z_cam = float(tvec[2][0])   # forward distance (meters)
        x_cam = float(tvec[0][0])
        y_cam = float(tvec[1][0])

        # ========================================================
        # Convert to MAVROS ENU
        # ENU: X forward, Y right, Z down
        # ========================================================
        x = z_cam
        y = x_cam
        z = y_cam

        # ========================================================
        # Orientation
        # ========================================================
        R, _ = cv2.Rodrigues(rvec)

        yaw = math.atan2(R[1, 0], R[0, 0])
        pitch = math.asin(-R[2, 0])
        roll = math.atan2(R[2, 1], R[2, 2])

        # ========================================================
        # Publish vision pose
        # ========================================================
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "map"

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z

        q = self.euler_to_quaternion(roll, pitch, yaw)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        self.vision_pub.publish(pose)

        # ========================================================
        # Publish DEPTH SENSOR (this unlocks GUIDED)
        # MUST be POSITIVE meters
        # ========================================================
        depth = Range()
        depth.header.stamp = pose.header.stamp
        depth.header.frame_id = "base_link"

        depth.radiation_type = Range.INFRARED
        depth.field_of_view = 0.1
        depth.min_range = 0.2
        depth.max_range = 10.0

        depth.range = np.clip(abs(z_cam), 0.3, 10.0)

        self.depth_pub.publish(depth)

        self.show_view(color, tag)

    # ============================================================
    def euler_to_quaternion(self, roll, pitch, yaw):
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

    # ============================================================
    def show_view(self, image, tag):
        if tag is not None:
            corners = tag.corners.astype(int)
            for i in range(4):
                cv2.line(
                    image,
                    tuple(corners[i]),
                    tuple(corners[(i + 1) % 4]),
                    (0, 255, 0),
                    2
                )
            c = tuple(tag.center.astype(int))
            cv2.circle(image, c, 5, (0, 0, 255), -1)

        cv2.imshow("AprilTag PnP + Depth", image)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = AprilTagPnPVision()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
