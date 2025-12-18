#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped, PoseStamped, Twist

QGC_IP = "192.168.4.1"
QGC_PORT = 5600

class StreamDisplayNode(Node):
    def __init__(self):
        super().__init__("stream_display_node")
        self.get_logger().info("Initializing Stream Node...")
        self.bridge = CvBridge()

        # --- Subscriptions ---
        # Ensure these topic names exactly match 'ros2 topic list'
        self.img_sub = self.create_subscription(Image, "/camera/color/image_raw", self.img_cb, 10)
        self.corners_sub = self.create_subscription(PolygonStamped, "/apriltag/corners", self.corners_cb, 10)
        self.vel_sub = self.create_subscription(Twist, "/mavros/setpoint_velocity/cmd_vel_unstamped", self.vel_cb, 10)
        self.pose_sub = self.create_subscription(PoseStamped, "/mavros/vision_pose/pose", self.pose_cb, 10)

        # --- Robust GStreamer Pipeline ---
        # Added 'sync=false' to udpsink to prevent hanging if the network buffer is full
        gst_pipeline = (
            f"appsrc ! videoconvert ! videoscale ! "
            f"video/x-raw,width=640,height=480,format=I420 ! "
            f"x264enc tune=zerolatency bitrate=800 speed-preset=ultrafast ! "
            f"rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={QGC_IP} port={QGC_PORT} sync=false"
        )
        
        self.get_logger().info(f"Opening GStreamer: {QGC_IP}:{QGC_PORT}")
        self.video_writer = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, 30.0, (640, 480), True)

        if not self.video_writer.isOpened():
            self.get_logger().error("CRITICAL: GStreamer Pipeline failed! Check your GStreamer plugins.")
        else:
            self.get_logger().info("GStreamer Pipeline opened successfully.")

        self.tag_pts = None
        self.current_vel = None
        self.current_pose = None
        self.frame_count = 0

    def corners_cb(self, msg): self.tag_pts = msg
    def vel_cb(self, msg): self.current_vel = msg
    def pose_cb(self, msg): self.current_pose = msg

    def img_cb(self, msg):
        # Debugging: print every 30th frame to console so you know the node is alive
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.get_logger().info(f"Processing frame {self.frame_count}...")

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return
        
        # --- Handle Coordinate Scaling ---
        orig_h, orig_w = frame.shape[:2]
        frame_resized = cv2.resize(frame, (640, 480))
        scale_x = 640.0 / orig_w
        scale_y = 480.0 / orig_h

        if self.tag_pts:
            pts = np.array([[p.x * scale_x, p.y * scale_y] for p in self.tag_pts.polygon.points], np.int32)
            cv2.polylines(frame_resized, [pts], True, (0, 255, 0), 2)

        # --- Overlay Telemetry ---
        if self.current_pose:
            p = self.current_pose.pose.position
            cv2.putText(frame_resized, f"Z_Dist: {p.x:.2f}m", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Push to GStreamer
        self.video_writer.write(frame_resized)

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
