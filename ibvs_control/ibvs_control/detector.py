#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import pyrealsense2 as rs
from cv_bridge import CvBridge
from pupil_apriltags import Detector
from geometry_msgs.msg import PolygonStamped, Point32
from sensor_msgs.msg import Image

from ibvs_control.constants import FX, FY, CX, CY, DIST_COEFFS

class AprilTagDetectorNode(Node):
    def __init__(self):
        super().__init__("apriltag_detector")

        # Publishers
        self.corners_pub = self.create_publisher(PolygonStamped, "/apriltag/corners", 10)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", 10)

        self.bridge = CvBridge()

        # --- RealSense Setup ---
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        # Using 1280x720 to match your base code resolution
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        
        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)

        # Filters for stable depth (Crucial for IBVS)
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

        # --- AprilTag Detector Setup ---
        self.detector = Detector(
            families="tag36h11",
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=True,
        )

        self.timer = self.create_timer(0.01, self.loop) # 100Hz attempts
        self.get_logger().info("Precision AprilTag detector node started")

    def order_corners_ccw(self, pts):
        """ Ensures corners are always in the same order to prevent IBVS sign-flip """
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])
        idx = np.argsort(angles)
        pts = pts[idx]
        # Rotate so top-left (min sum of x+y) is first
        s = pts.sum(axis=1)
        start = np.argmin(s)
        return np.roll(pts, -start, axis=0)

    def loop(self):
        # Use poll_for_frames to prevent blocking the ROS executor
        frames = self.pipeline.poll_for_frames()
        if not frames:
            return

        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            return

        # Apply Filters
        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)

        # Convert to numpy
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        detections = self.detector.detect(gray)

        if detections:
            # We target the first detection or a specific ID
            tag = detections[0]
            tag = detections[0]
            raw_pts = self.order_corners_ccw(tag.corners.astype(np.float32))

            # --- NEW: UNDISTORT CORNERS ---
            # This converts raw 'distorted' pixels into 'ideal' pinhole pixels
            camera_matrix = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
            dist_coeffs = np.array(DIST_COEFFS)
            
            # undistortPoints expects shape (N, 1, 2)
            pts_reshaped = raw_pts.reshape(-1, 1, 2)
            undistorted_pts = cv2.undistortPoints(
                pts_reshaped, camera_matrix, dist_coeffs, 
                R=None, P=camera_matrix
            )
            pts = undistorted_pts.reshape(-1, 2)
            # ------------------------------

            msg = PolygonStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_color_optical_frame"

            for (u, v) in pts:
                p = Point32()
                p.x = float(u)
                p.y = float(v)
                p.z = 0.0 # Standard for PolygonStamped
                msg.polygon.points.append(p)

            self.corners_pub.publish(msg)

        # Always publish depth to keep the IBVS callback synced
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="16UC1")
        depth_msg.header.stamp = self.get_clock().now().to_msg()
        depth_msg.header.frame_id = "camera_depth_optical_frame"
        self.depth_pub.publish(depth_msg)

def main():
    rclpy.init()
    node = AprilTagDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pipeline.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
