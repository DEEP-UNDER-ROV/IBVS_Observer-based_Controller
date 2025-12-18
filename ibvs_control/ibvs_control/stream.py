#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped, PoseStamped, Twist

# Importing your shared constants
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

        # --- GStreamer Writer Config ---
        # Using 640x480 for the stream to match the typical detection resolution 
        # and ensure low-latency performance over UDP.
        STREAM_WIDTH = 1280
        STREAM_HEIGHT = 720
        
        # Updated pipeline with explicit format negotiation (I420) and zerolatency tuning
        gst_pipeline = (
            f"appsrc ! videoconvert ! videoscale ! "
            f"video/x-raw,format=I420,width={STREAM_WIDTH},height={STREAM_HEIGHT} ! "
            f"x264enc tune=zerolatency bitrate=800 speed-preset=ultrafast ! "
            f"rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={QGC_IP} port={QGC_PORT}"
        )
        
        self.video_writer = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, 30.0, (STREAM_WIDTH, STREAM_HEIGHT), True)

        if not self.video_writer.isOpened():
            self.get_logger().error("GStreamer Pipeline failed to open! Check x264 plugins.")

        # Storage
        self.tag_pts = None
        self.current_vel = None
        self.current_pose = None

        self.get_logger().info(f"Streaming to QGC at {QGC_IP}:{QGC_PORT}")

    def corners_cb(self, msg): self.tag_pts = msg
    def vel_cb(self, msg): self.current_vel = msg
    def pose_cb(self, msg): self.current_pose = msg

    def img_cb(self, msg):
        # 1. Convert ROS image to CV2
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Ensure frame matches STREAM dimensions
        if frame.shape[1] != 640 or frame.shape[0] != 480:
            frame = cv2.resize(frame, (640, 480))

        # 2. Draw AprilTag Corners
        if self.tag_pts:
            pts = np.array([[p.x, p.y] for p in self.tag_pts.polygon.points], np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            for i, p in enumerate(pts):
                cv2.circle(frame, tuple(p), 5, (255, 0, 0), -1)
                cv2.putText(frame, str(i+1), (p[0]+8, p[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # 3. Draw PnP Pose Overlay
        if self.current_pose:
            p = self.current_pose.pose.position
            p_text = f"PnP Pose: X:{p.x:.2f} Y:{p.y:.2f} Z:{p.z:.2f}"
            cv2.putText(frame, p_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # 4. Draw Body Velocity Overlay
        if self.current_vel:
            v = self.current_vel
            v_text = f"Vb: x:{v.linear.x:+.2f} y:{v.linear.y:+.2f} z:{v.linear.z:+.2f}"
            w_text = f"Wb: yaw:{v.angular.z:+.2f}"
            cv2.putText(frame, v_text, (20, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, w_text, (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 5. Push to GStreamer
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
