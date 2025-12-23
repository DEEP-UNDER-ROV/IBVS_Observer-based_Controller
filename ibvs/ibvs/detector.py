#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import pyrealsense2 as rs

from cv_bridge import CvBridge
from pupil_apriltags import Detector

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PolygonStamped, Point32, PoseStamped, Twist, Vector3Stamped
from sensor_msgs.msg import Image

from ibvs.constants import *

class IBVS_Telemetry(Node):
    def desired_corners_from_Z(self, Z_des):
        half = TAG_SIZE / 2.0
    
        # Tag corners in camera frame (meters)
        corners_3d = np.array([
            [-half, -half, Z_des],
            [ half, -half, Z_des],
            [ half,  half, Z_des],
            [-half,  half, Z_des],
        ])
    
        desired = np.zeros((4, 2), dtype=np.float32)
    
        for i, (X, Y, Z) in enumerate(corners_3d):
            u = FX * (X / Z) + CX
            v = FY * (Y / Z) + CY
            desired[i] = [u, v]
    
        return desired
    
    def __init__(self):
        super().__init__("IBVS_Telemetry")
        self.get_logger().info("Telemetry Node Started")

        # --- ROS Publishers ---
        self.corners_pub = self.create_publisher(PolygonStamped, "/apriltag/corners", 10)
        self.img_sub = self.create_subscription(Image, "/camera/color/image_raw", self.img_cb, 10)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", 10)

        # --- ROS Subscriptions ---
        
        self.pose_sub = self.create_subscription(PoseStamped, "/mavros/vision_pose/pose", self.pose_cb, 10)        
        self.pos_sub = self.create_publisher(Point, "/ibvs/pos", self.pos_cb, 10)      
        self.vel_sub = self.create_subscription(Twist, "/ibvs/vel", self.vel_cb, 10)
        self.err_sub = self.create_subscription(Float32MultiArray, "/ibvs/error", self.err_cb, 10)
        

        self.current_vel = None
        self.current_pos = None
        self.current_pose = None
        self.current_err = None

        self.camera_matrix = np.array([
            [FX, 0, CX],
            [0, FY, CY],
            [0,  0,  1]
        ], dtype=np.float32)

        self.dist_coeffs = np.array(DIST_COEFFS, dtype=np.float32)

        # --- RealSense Setup (High Res for Accuracy) ---
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.pipeline.start(cfg)
        
        self.align = rs.align(rs.stream.color)
        self.bridge = CvBridge()
        
        # --- AprilTag Detector Setup ---
        self.detector = Detector(families="tag36h11", nthreads=4, quad_decimate=1.0, refine_edges=True)

        # --- GStreamer VideoWriter (Low Res for Streaming) ---
        # Note: Added 'sync=false' and 'videoscale' to ensure 480p output
        gst_pipeline = (
            f"appsrc ! videoconvert ! "
            f"video/x-raw,width=640,height=480,format=I420 ! "
            f"x264enc tune=zerolatency bitrate=1000 speed-preset=ultrafast ! "
            f"rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={QGC_IP} port={QGC_PORT} sync=false"
        )
        self.video_writer = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, 30, (640, 480), True)

        if not self.video_writer.isOpened():
            self.get_logger().error("GStreamer failed! Check: gst-inspect-1.0 x264enc")

        self.timer = self.create_timer(0.033, self.loop)
        self.get_logger().info(f"Streaming to QGC at {QGC_IP}:{QGC_PORT}")

    def vel_cb(self, msg): self.current_vel = msg
    def pos_cb(self, msg): self.current_pos = msg
    def pose_cb(self, msg): self.current_pose = msg
    def err_cb(self, msg): self.current_err = msg

    @staticmethod
    def order_corners_apriltag(corners):
        return np.array([
            corners[0],  # TL
            corners[1],  # TR
            corners[2],  # BR
            corners[3],  # BL
        ], dtype=np.float32)

    def loop(self):
        frames = self.pipeline.poll_for_frames()
        if not frames:
            return

        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        
        # 1. Detect Tags (at 1280x720)
        detections = self.detector.detect(gray)
        undistorted_pts = None
        
        if detections:
            tag = detections[0]
            raw_pts = self.order_corners_apriltag(tag.corners)
            pts = raw_pts.reshape(-1, 1, 2)
            undistorted = cv2.undistortPoints(pts, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)          
            undistorted_pts = undistorted.reshape(-1, 2)

            # Publish corners for the IBVS node
            poly = PolygonStamped()
            poly.header.stamp = self.get_clock().now().to_msg()
            poly.header.frame_id = "camera_color_optical_frame"
            
            for (u, v) in undistorted_pts:
                p = Point32()
                p.x, p.y, p.z = float(u), float(v), 0.0
                poly.polygon.points.append(p)
                
            self.corners_pub.publish(poly)

        # 2. Draw & Stream (Downscale to 640x480)
        stream = cv2.resize(color, (640, 480))
        sx, sy = 640 / 1280, 480 / 720
        
        # Use polylines for the red box (cleaner and avoids manual indexing)
        if undistorted_pts is not None:
            pts_u = np.asarray(undistorted_pts).reshape(4, 2)
            pts_u_draw = (pts_u * [sx, sy]).astype(np.int32)

            cv2.polylines(stream, [pts_u_draw], True, (0, 255, 0), 2)
        
        for i, (u, v) in enumerate(pts_u_draw):
                cv2.circle(stream, (u, v), 5, (255, 0, 0), -1)
                cv2.putText(stream, f"{i+1}", (u+6, v-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if raw_pts is not None:
            pts_r = np.asarray(raw_pts).reshape(4, 2)
            pts_r_draw = (pts_r * [sx, sy]).astype(np.int32)
        
            cv2.polylines(stream, [pts_r_draw], True, (0, 180, 180), 1)
        
        if self.current_vel is not None:
            v = self.current_vel
            cv2.putText(stream, f"Vx:{v.linear.x:+.2f}", (20,25), 2,  0.5, (0,255,255), 2)
            cv2.putText(stream, f"Vy:{v.linear.y:+.2f}", (20,45), 2,  0.5, (0,255,255), 2)
            cv2.putText(stream, f"Vz:{v.linear.z:+.2f}", (20,65), 2,  0.5, (0,255,255), 2)
            cv2.putText(stream, f"Wx:{v.angular.x:+.2f}", (20,85), 2, 0.5, (0,200,255), 2)
            cv2.putText(stream, f"Wy:{v.angular.y:+.2f}", (20,105), 2, 0.5, (0,200,255), 2)
            cv2.putText(stream, f"Wz:{v.angular.z:+.2f}", (20,125), 2, 0.5, (0,200,255), 2)

        # ---------------- Error Overlay ----------------
        if self.current_err is not None:
            e = np.asarray(self.current_err.data)
            
            x0, y0, dy = 145, 25, 20
            for i in range(4):
                ex, ey, ez = e[i*3:(i+1)*3]
                cv2.putText(stream, f"P{i+1}: ex={ex:+.2f}px  ey={ey:+.2f}px  ez={ez:+.2f} m",
                    (x0, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Push to QGC
        self.video_writer.write(stream)

        # Publish Depth Map
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="16UC1")
        depth_msg.header.stamp = self.get_clock().now().to_msg()
        self.depth_pub.publish(depth_msg)

def main():
    rclpy.init()
    node = IBVS_Telemetry()
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
