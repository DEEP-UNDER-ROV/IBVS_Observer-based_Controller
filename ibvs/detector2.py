#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import pyrealsense2 as rs

from geometry_msgs.msg import PolygonStamped, Point32
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

from pupil_apriltags import Detector
from ibvs.constants import *


PATCH = 3   # depth patch radius (3 → 7×7)


class IBVSDetectorDepth(Node):
    def __init__(self):
        super().__init__("ibvs_detector_depth")

        # ---------------- Parameters ----------------
        self.declare_parameter("detect_fps", 10.0)
        self.declare_parameter("image_pub_fps", 20.0)
        self.declare_parameter("publish_images", False)

        self.detect_period = 1.0 / self.get_parameter("detect_fps").value
        self.image_pub_period = 1.0 / self.get_parameter("image_pub_fps").value

        # ---------------- Publishers ----------------
        self.corners_pub = self.create_publisher(
            PolygonStamped, "/apriltag/corners_depth", 10
        )
        self.raw_pub = self.create_publisher(
            Image, "/camera/color/image_raw", 10
        )
        self.comp_pub = self.create_publisher(
            CompressedImage, "/camera/color/image_raw/compressed", 10
        )

        self.bridge = CvBridge()

        # ---------------- RealSense ----------------
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.pipeline.start(cfg)

        self.align = rs.align(rs.stream.color)

        # ---------------- AprilTag ----------------
        self.detector = Detector(
            families="tag36h11",
            nthreads=2,
            quad_decimate=1.0,
            refine_edges=True
        )

        # ---------------- Timing ----------------
        self.last_detect = 0.0
        self.last_img_pub = 0.0

        self.timer = self.create_timer(1.0 / 30.0, self.loop)

        self.get_logger().info("IBVS detector + depth node started")

    # -------------------------------------------------

    @staticmethod
    def order_corners(c):
        return np.array([c[3], c[2], c[1], c[0]], dtype=np.float32)

    # -------------------------------------------------

    def sample_depth(self, depth, u, v):
        h, w = depth.shape
        ui, vi = int(u), int(v)

        if not (0 <= ui < w and 0 <= vi < h):
            return None

        patch = depth[
            max(0, vi-PATCH):min(h, vi+PATCH+1),
            max(0, ui-PATCH):min(w, ui+PATCH+1)
        ]

        valid = patch[patch > 0]
        if valid.size == 0:
            return None

        return float(np.median(valid)) * 0.001  # mm → m

    # -------------------------------------------------

    def loop(self):
        frames = self.pipeline.poll_for_frames()
        if not frames:
            return

        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        # ---------------- Detection @ 10 Hz ----------------
        if now - self.last_detect >= self.detect_period:
            self.last_detect = now

            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (640, 360), interpolation=cv2.INTER_AREA)

            detections = self.detector.detect(gray_small)

            if detections:
                tag = detections[0]
                pts = self.order_corners(tag.corners)

                # scale back to 1280×720
                pts[:, 0] *= 2.0
                pts[:, 1] *= 2.0

                poly = PolygonStamped()
                poly.header.stamp = self.get_clock().now().to_msg()
                poly.header.frame_id = "camera_color_optical_frame"

                for (u, v) in pts:
                    Z = self.sample_depth(depth, u, v)
                    if Z is None:
                        return

                    p = Point32(
                        x=float(u),
                        y=float(v),
                        z=Z
                    )
                    poly.polygon.points.append(p)

                self.corners_pub.publish(poly)

        # ---------------- Optional image publishing ----------------
        if not self.get_parameter("publish_images").value:
            return

        if now - self.last_img_pub < self.image_pub_period:
            return

        self.last_img_pub = now
        stamp = self.get_clock().now().to_msg()

        raw = self.bridge.cv2_to_imgmsg(color, encoding="bgr8")
        raw.header.stamp = stamp
        raw.header.frame_id = "camera_link"
        self.raw_pub.publish(raw)

        comp = CompressedImage()
        comp.header = raw.header
        comp.format = "jpeg"
        comp.data = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        self.comp_pub.publish(comp)

    # -------------------------------------------------

    def destroy_node(self):
        self.pipeline.stop()
        super().destroy_node()


def main():
    rclpy.init()
    node = IBVSDetectorDepth()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
