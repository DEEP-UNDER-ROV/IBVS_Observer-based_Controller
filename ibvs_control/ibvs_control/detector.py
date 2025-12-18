#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import pyrealsense2 as rs
from cv_bridge import CvBridge
from pupil_apriltags import Detector
from geometry_msgs.msg import PolygonStamped, Point32, PoseStamped, Twist
from sensor_msgs.msg import Image

from ibvs_control.constants import FX, FY, CX, CY, DIST_COEFFS, DESIRED_SIZE

# --- Streaming Config ---
QGC_IP = "192.168.4.1"
QGC_PORT = 5600

class AprilTagUnifiedNode(Node):
    def __init__(self):
        super().__init__("apriltag_unified_node")
        self.get_logger().info("Initializing Detector node")

        # --- ROS Publishers ---
        self.corners_pub = self.create_publisher(PolygonStamped, "/apriltag/corners", 10)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", 10)
        
        # --- ROS Subscriptions (for Overlay Telemetry) ---
        self.vel_sub = self.create_subscription(Twist, "/mavros/setpoint_velocity/cmd_vel_unstamped", self.vel_cb, 10)
        self.pose_sub = self.create_subscription(PoseStamped, "/mavros/vision_pose/pose", self.pose_cb, 10)

        self.bridge = CvBridge()
        self.current_vel = None
        self.current_pose = None

        # --- RealSense Setup (High Res for Accuracy) ---
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        
        IMG_W, IMG_H = 1280, 720
        S = DESIRED_SIZE  # 280 px
        
        cx, cy = IMG_W / 2, IMG_H / 2
        
        self.desired = np.array([
            [cx - S/2, cy - S/2],  # top-left
            [cx + S/2, cy - S/2],  # top-right
            [cx + S/2, cy + S/2],  # bottom-right
            [cx - S/2, cy + S/2],  # bottom-left
        ], dtype=np.float32)

        # --- AprilTag Detector Setup ---
        self.detector = Detector(families="tag36h11", nthreads=4, quad_decimate=1.0, refine_edges=True)

        # --- GStreamer VideoWriter (Low Res for Streaming) ---
        # Note: Added 'sync=false' and 'videoscale' to ensure 480p output
        gst_pipeline = (
            f"appsrc ! videoconvert ! videoscale ! "
            f"video/x-raw,width=640,height=480,format=I420 ! "
            f"x264enc tune=zerolatency bitrate=600 speed-preset=ultrafast ! "
            f"rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={QGC_IP} port={QGC_PORT} sync=false"
        )
        self.video_writer = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, 30, (640, 480), True)

        if not self.video_writer.isOpened():
            self.get_logger().error("GStreamer failed! Check: gst-inspect-1.0 x264enc")

        self.timer = self.create_timer(0.033, self.loop)
        self.get_logger().info(f"Streaming to QGC at {QGC_IP}:{QGC_PORT} (640x480)")

    def vel_cb(self, msg): self.current_vel = msg
    def pose_cb(self, msg): self.current_pose = msg
    def err_cb(self, msg): self.current_err = msg

    def order_corners_ccw(self, pts):
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])
        idx = np.argsort(angles)
        pts = pts[idx]
        s = pts.sum(axis=1)
        return np.roll(pts, -np.argmin(s), axis=0)

    def loop(self):
        frames = self.pipeline.poll_for_frames()
        if not frames:
            return

        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        
        # 1. Detect Tags (at 1280x720)
        detections = self.detector.detect(gray)
        raw_pts = None
        
        if detections:
            tag = detections[0]
            raw_pts = self.order_corners_ccw(tag.corners.astype(np.float32))

            # Undistort (Precision IBVS)
            camera_matrix = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
            dist_coeffs = np.array(DIST_COEFFS)
            pts_reshaped = raw_pts.reshape(-1, 1, 2)
            undistorted_pts = cv2.undistortPoints(pts_reshaped, camera_matrix, dist_coeffs, R=None, P=camera_matrix)
            tag_pts_to_publish = undistorted_pts.reshape(-1, 2)

            # Publish corners for the IBVS node
            poly_msg = PolygonStamped()
            poly_msg.header.stamp = self.get_clock().now().to_msg()
            poly_msg.header.frame_id = "camera_color_optical_frame"
            for (u, v) in tag_pts_to_publish:
                p = Point32()
                p.x, p.y = float(u), float(v)
                poly_msg.polygon.points.append(p)
            self.corners_pub.publish(poly_msg)

        # 2. Draw & Stream (Downscale to 640x480)
        stream_frame = cv2.resize(color, (640, 480))
        scale_x, scale_y = 640/1280, 480/720

        desired_scaled = (self.desired * [scale_x, scale_y]).astype(np.int32)
        
        # Use polylines for the red box (cleaner and avoids manual indexing)
        cv2.polylines(stream_frame, [desired_scaled], True, (0, 0, 255), 2)
        
        for i, (x, y) in enumerate(desired_scaled):
            cv2.circle(stream_frame, (int(x), int(y)), 4, (0, 0, 255), -1)
            cv2.putText(stream_frame, f"D{i+1}", (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1 )

        # Draw scaled tag
        if detections:
            pts_scaled = (raw_pts * [scale_x, scale_y]).astype(np.int32)
            cv2.polylines(stream_frame, [pts_scaled], True, (0, 255, 0), 2)
            
            for i, (u, v) in enumerate(pts_scaled):
                cv2.circle(stream_frame, (int(u), int(v)), 5, (255, 0, 0), -1)
                cv2.putText(stream_frame, f"{i+1}", (int(u)+6, int(v)-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if self.current_vel:
            v = self.current_vel
            cv2.putText(frame, f"Vx:{v.linear.x:+.2f}", (10,20), 2, 0.6, (0,255,255),2)
            cv2.putText(frame, f"Vy:{v.linear.y:+.2f}", (10,45), 2, 0.6, (0,255,255),2)
            cv2.putText(frame, f"Vz:{v.linear.z:+.2f}", (10,70), 2, 0.6, (0,255,255),2)
            cv2.putText(frame, f"Wz:{v.angular.z:+.2f}", (10,95), 2, 0.6, (0,200,255),2)

        # ---------------- Error Overlay ----------------
        if self.current_err:
            e = self.current_err.vector
            norm = np.linalg.norm([e.x, e.y, e.z])
            cv2.putText(frame, f"Err u:{e.x:+.4f}", (430,20),2,0.6,(255,100,100),2)
            cv2.putText(frame, f"Err v:{e.y:+.4f}", (430,45),2,0.6,(255,100,100),2)
            cv2.putText(frame, f"Err z:{e.z:+.3f}", (430,70),2,0.6,(255,100,100),2)
            cv2.putText(frame, f"||e||:{norm:.4f}", (430,95),2,0.6,(0,0,255),2)

        # Push to QGC
        self.video_writer.write(stream_frame)

        # Publish Depth Map
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="16UC1")
        depth_msg.header.stamp = self.get_clock().now().to_msg()
        self.depth_pub.publish(depth_msg)

def main():
    rclpy.init()
    node = AprilTagUnifiedNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.video_writer.release()
        node.pipeline.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
