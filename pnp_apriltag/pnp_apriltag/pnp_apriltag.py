#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import cv2
import math
import pyrealsense2 as rs

from pupil_apriltags import Detector
from geometry_msgs.msg import PoseStamped


# =========================
# QGC VIDEO SETTINGS
# =========================
QGC_IP = "192.168.4.1"
QGC_PORT = 5600
VIDEO_FPS = 30
VIDEO_SIZE = (640, 480)


class AprilTagPnPVision(Node):

    def __init__(self):
        super().__init__('apriltag_pnp_vision')

        # ======================================================
        # MAVROS Vision Pose publisher
        # ======================================================
        self.vision_pub = self.create_publisher(
            PoseStamped,
            '/mavros/vision_pose/pose',
            10
        )

        # ======================================================
        # GStreamer → QGC (UDP H.264)
        # ======================================================
        gst_pipeline = (
            "appsrc ! videoconvert ! "
            "x264enc tune=zerolatency bitrate=500 speed-preset=ultrafast ! "
            "rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={QGC_IP} port={QGC_PORT}"
        )

        self.video = cv2.VideoWriter(
            gst_pipeline,
            cv2.CAP_GSTREAMER,
            0,
            VIDEO_FPS,
            VIDEO_SIZE,
            True
        )

        if not self.video.isOpened():
            self.get_logger().error("Failed to open GStreamer video stream")
        else:
            self.get_logger().info(
                f"Streaming video to QGC at {QGC_IP}:{QGC_PORT}"
            )

        # ======================================================
        # RealSense pipeline (COLOR ONLY)
        # ======================================================
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color,
            VIDEO_SIZE[0],
            VIDEO_SIZE[1],
            rs.format.bgr8,
            VIDEO_FPS
        )
        profile = self.pipeline.start(config)

        # ======================================================
        # Camera intrinsics
        # ======================================================
        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()

        self.camera_matrix = np.array([
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        # ======================================================
        # AprilTag detector
        # ======================================================
        self.detector = Detector(
            families="tag36h11",
            nthreads=2,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=True,
        )

        # ======================================================
        # Tag geometry (meters)
        # ======================================================
        self.tag_size = 0.16  # <-- SET YOUR TAG SIZE
        s = self.tag_size / 2.0

        self.object_points = np.array([
            [-s, -s, 0.0],
            [ s, -s, 0.0],
            [ s,  s, 0.0],
            [-s,  s, 0.0],
        ], dtype=np.float32)

        # ======================================================
        # Timer (20 Hz)
        # ======================================================
        self.timer = self.create_timer(0.05, self.loop)

        self.get_logger().info(
            "AprilTag PnP vision + UDP video streaming started"
        )

    # ==========================================================
    def loop(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return

        frame = np.asanyarray(color_frame.get_data())

        # ======================================================
        # Stream raw video to QGC
        # ======================================================
        if self.video.isOpened():
            self.video.write(frame)

        # ======================================================
        # AprilTag detection
        # ======================================================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)

        if len(detections) == 0:
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
            return

        # ======================================================
        # OpenCV camera frame:
        # X right, Y down, Z forward
        # ======================================================
        z_cam = float(tvec[2][0])
        x_cam = float(tvec[0][0])
        y_cam = float(tvec[1][0])

        # ======================================================
        # Convert to MAVROS ENU
        # ENU: X forward, Y right, Z down
        # ======================================================
        x = z_cam
        y = x_cam
        z = y_cam

        # ======================================================
        # Orientation
        # ======================================================
        R, _ = cv2.Rodrigues(rvec)
        yaw = math.atan2(R[1, 0], R[0, 0])
        pitch = math.asin(-R[2, 0])
        roll = math.atan2(R[2, 1], R[2, 2])

        # ======================================================
        # Publish vision pose
        # ======================================================
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

    # ==========================================================
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
    node = AprilTagPnPVision()
    rclpy.spin(node)
    node.video.release()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
