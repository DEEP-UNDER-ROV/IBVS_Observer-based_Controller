#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image

from ibvs.constants import *


class CleanVideoStreamer(Node):
    def __init__(self):
        super().__init__("stream")
        
        # --- Parameter Setup for Dynamic Topic Selection ---
        # Default topic can be easily switched to IR or RGB via runtime arguments
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.target_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        
        self.get_logger().info(f"Stream Initiated. Subscribed to: {self.target_topic}")

        self.bridge = CvBridge()

        # --- GStreamer Pipeline Configuration ---
        gst_pipeline = (
            f"appsrc is-live=true block=true do-timestamp=true format=time ! "
            f"queue ! "
            f"videoconvert ! "
            f"video/x-raw,width={stream_w},height={stream_h},format=I420 ! "
            f"x264enc tune=zerolatency "
            f"bitrate=2000 speed-preset=ultrafast "
            f"key-int-max=30 ! "
            f"rtph264pay config-interval=-1 pt=96 ! "
            f"udpsink host={QGC_IP} port={QGC_PORT} sync=false async=false"
        )
        
        self.video_writer = cv2.VideoWriter(
            gst_pipeline, cv2.CAP_GSTREAMER, 0, 30, (stream_w, stream_h), True
        )

        if not self.video_writer.isOpened():
            self.get_logger().error("GStreamer pipeline initialization failed!")
        else:
            self.get_logger().info(f"Streaming H.264 video cleanly to {QGC_IP}:{QGC_PORT}")

        # --- Subscription ---
        self.create_subscription(Image, self.target_topic, self.cb_image, 10)

    def cb_image(self, msg):
        try:
            # Handle mono/IR streams and standard RGB streams seamlessly
            if "infra" in self.target_topic:
                # IR streams are single-channel (mono8); convert to bgr8 for GStreamer conversion pipeline
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Failed to convert frame: {e}")
            return

        h, w = frame.shape[:2]

        # Enforce exact bounding constraints for the underlying x264 video stream
        if (h, w) != (stream_w, stream_h):
            gst_frame = cv2.resize(frame, (stream_w, stream_h), interpolation=cv2.INTER_LINEAR)
        else:
            gst_frame = frame

        # Push the un-overlayed raw frame directly out via RTP
        if self.video_writer.isOpened():
            self.video_writer.write(gst_frame)

    def destroy_node(self):
        if hasattr(self, 'video_writer') and self.video_writer.isOpened():
            self.video_writer.release()
            self.get_logger().info("GStreamer Pipeline released.")
        super().destroy_node()


def main():
    rclpy.init()
    node = CleanVideoStreamer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()