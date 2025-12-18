#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped, PoseStamped, Twist
from ibvs_control.constants import FX, FY, CX, CY

QGC_IP = "192.168.4.1"
QGC_PORT = 5600

class StreamDisplayNode(Node):
    def __init__(self):
        super().__init__("stream_display_node")
        self.bridge = CvBridge()

        # --- Subscriptions ---
        self.img_sub = self.create_subscription(Image, "/camera/color/image_raw", self.img_cb, 10)
        self.corners_sub = self.create_subscription(PolygonStamped, "/apriltag/corners", self.corners_cb, 10)
        self.vel_sub = self.create_subscription(Twist, "/mavros/setpoint_velocity/cmd_vel_unstamped", self.vel_cb, 10)
        self.pose_sub = self.create_subscription(PoseStamped, "/mavros/vision_pose/pose", self.pose_cb, 10)

        # --- Robust GStreamer Pipeline for QGC ---
        # 1. videoconvert: ensures BGR to YUV conversion
        # 2. videoscale: forces frame size to 640x480
        # 3. x264enc: high speed, low latency
        gst_pipeline = (
            f"appsrc ! videoconvert ! videoscale ! "
            f"video/x-raw,width=640,height=480,format=I420 ! "
            f"x264enc tune=zerolatency bitrate=800 speed-preset=ultrafast ! "
            f"rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={QGC_IP} port={QGC_PORT}"
        )
        
        self.video_writer = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, 30.0, (640, 480), True)

        if not self.video_writer.isOpened():
            self.get_logger().error("GStreamer Pipeline failed! Check if 'gst-inspect-1.0 x264enc' works.")

        self.tag_pts = None
        self.current_vel = None
        self.current_pose = None

    def corners_cb(self, msg): self.tag_pts = msg
    def vel_cb(self, msg): self.current_vel = msg
    def pose_cb(self, msg): self.current_pose = msg

    def img_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # --- Handle Coordinate Scaling ---
        # If detector is 1280x720, we must scale points to fit 640x480 drawing
        orig_h, orig_w = frame.shape[:2]
        frame = cv2.resize(frame, (640, 480))
        scale_x = 640.0 / orig_w
        scale_y = 480.0 / orig_h

        if self.tag_pts:
            # Scale corners so they align with resized image
            pts = np.array([[p.x * scale_x, p.y * scale_y] for p in self.tag_pts.polygon.points], np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            for i, p in enumerate(pts):
                cv2.circle(frame, tuple(p), 5, (255, 0, 0), -1)

        # --- Overlay Telemetry ---
        if self.current_pose:
            p = self.current_pose.pose.position
            cv2.putText(frame, f"PnP: {p.x:.2f}, {p.y:.2f}, {p.z:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        if self.current_vel:
            v = self.current_vel.linear
            cv2.putText(frame, f"Cmd Vel: {v.x:+.2f} {v.y:+.2f} {v.z:+.2f}", (10, 460), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        self.video_writer.write(frame)

def main():
    rclpy.init()
    node = StreamDisplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.video_writer:
            node.video_writer.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
