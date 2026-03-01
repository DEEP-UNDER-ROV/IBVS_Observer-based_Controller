#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import pyrealsense2 as rs

from cv_bridge import CvBridge
from pupil_apriltags import Detector

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PolygonStamped, Point32, Twist, Point, Vector3Stamped, TwistStamped
from sensor_msgs.msg import Image, CompressedImage
from mavros_msgs.msg import OverrideRCIn, RCOut

from ibvs.constants import *


class IBVS_Telemetry(Node):
    def desired_corners_from_Z(self, Z_des):
        half = TAG_SIZE / 2.0
        corners_3d = np.array([
            [-half, -half, Z_des],
            [ half, -half, Z_des],
            [ half,  half, Z_des],
            [-half,  half, Z_des],
        ])

        desired = np.zeros((4, 2), dtype=np.float32)
        for i, (X, Y, Z) in enumerate(corners_3d):
            desired[i] = [FX * X / Z + CX, FY * Y / Z + CY]
        return desired
    
    def reorder_corners_ccw(self, pts):
        pts = np.asarray(pts, dtype=np.float32)

        # compute centroid
        center = np.mean(pts, axis=0)

        # compute angle of each point w.r.t centroid
        angles = np.arctan2(pts[:,1] - center[1],
                            pts[:,0] - center[0])

        # sort by angle (CCW)
        sort_idx = np.argsort(angles)
        pts_sorted = pts[sort_idx]

        # ensure starting point is top-left
        s = pts_sorted.sum(axis=1)
        top_left_idx = np.argmin(s)
        pts_sorted = np.roll(pts_sorted, -top_left_idx, axis=0)

        return pts_sorted

    def __init__(self):
        super().__init__("IBVS_Telemetry")
        self.get_logger().info("Telemetry Node Started")

        # ---------------- Publishers ----------------
        self.corners_pub = self.create_publisher(PolygonStamped, "/apriltag/corners", 10)
        self.depth_pub   = self.create_publisher(Image, "/camera/depth/image_raw", 10)
        self.raw_pub     = self.create_publisher(Image, "/camera/color/image_raw", 10)
        self.comp_pub    = self.create_publisher(CompressedImage, "/camera/color/image_raw/compressed", 10)

        # ---------------- Subscriptions ----------------
        self.create_subscription(OverrideRCIn, "/mavros/rc/override", self.cb_rc, 10)
        self.create_subscription(RCOut, "/mavros/rc/out", self.cb_rc_out, 10)
        self.create_subscription(TwistStamped, "/ibvs/vel", self.cb_vel, 10)
        self.create_subscription(Float32MultiArray, "/ibvs/error", self.cb_err, 10)

        self.current_rc = None
        self.current_rc_out = None
        self.current_vel = None
        self.current_err = None
        self.last_poly = None

        self.camera_matrix = np.array([[FX, 0, CX],
                                       [0, FY, CY],
                                       [0,  0,  1]], dtype=np.float32)
        self.dist_coeffs = np.array(DIST_COEFFS, dtype=np.float32)
        self.desired = self.desired_corners_from_Z(Z_DES)

        # ---------------- RealSense ----------------
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

        self.bridge = CvBridge()

        # ---------------- AprilTag ----------------
        self.detector = Detector(
            families="tag36h11",
            nthreads=2,
            quad_decimate=0.8,
            quad_sigma=0.5,
            refine_edges=True
        )

        # ---------------- Efficiency controls ----------------
        self.detect_div = 1              # 30 Hz → 10 Hz detection
        self.frame_count = 0

        self.publish_full_depth = False  # enable only for rosbag
        self.img_pub_period = 1.0 / 20.0 # image @ 20 Hz
        self.last_img_pub = 0.0

        self.PATCH = 3

        # ---------------- Video Stream ----------------
        # gst_pipeline = (
        #     f"appsrc ! videoconvert ! "
        #     f"video/x-raw,width=640,height=480,format=I420 ! "
        #     f"x264enc tune=zerolatency bitrate=1000 speed-preset=ultrafast ! "
        #     f"rtph264pay config-interval=1 pt=96 ! "
        #     f"udpsink host={QGC_IP} port={QGC_PORT} sync=false"
        # )
        
        gst_pipeline = (
            f"appsrc is-live=true block=true do-timestamp=true format=time ! "
            f"queue ! "
            f"videoconvert ! "
            f"video/x-raw,width=640,height=480,format=I420 ! "
            f"x264enc tune=zerolatency "
            f"bitrate=1000 speed-preset=ultrafast "
            f"key-int-max=30 ! "
            f"rtph264pay config-interval=-1 pt=96 ! "
            f"udpsink host={QGC_IP} port={QGC_PORT} sync=false async=false"
        )
        
        self.video_writer = cv2.VideoWriter(
            gst_pipeline, cv2.CAP_GSTREAMER, 0, 30, (640, 480), True
        )

        self.timer = self.create_timer(1.0 / 30.0, self.loop)
        
        if not self.video_writer.isOpened():
            self.get_logger().error("GStreamer pipeline failed!")
            
        self.get_logger().info("IBVS Telemetry running")

    # ---------------- Callbacks ----------------
    def cb_rc(self, msg): self.current_rc = msg
    def cb_rc_out(self, msg): self.current_rc_out = msg
    def cb_vel(self, msg): self.current_vel = msg
    def cb_err(self, msg): self.current_err = msg

    def sample_depth(self, depth, u, v):
        h, w = depth.shape
        ui, vi = int(round(u)), int(round(v))

        if not (0 <= ui < w and 0 <= vi < h):
            return None

        # Primary: exact pixel depth
        center_depth = depth[vi, ui]

        if 50 < center_depth < 10000:   # valid range in mm
            return float(center_depth) * 0.001  # mm → m

        # Fallback: small neighborhood mean
        patch = depth[
            max(0, vi-self.PATCH):min(h, vi+self.PATCH+1),
            max(0, ui-self.PATCH):min(w, ui+self.PATCH+1)
        ]

        valid = patch[(patch > 0) & (patch < 10000)]
        if valid.size < 3:
            return None

        return float(np.mean(valid)) * 0.001

    # ---------------- Main loop ----------------
    def loop(self):
        frames = self.pipeline.wait_for_frames()
        if not frames:
            return

        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return

        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        stamp = self.get_clock().now().to_msg()

        self.frame_count += 1
        do_detect = (self.frame_count % self.detect_div) == 0

        undistorted_pts = None
        raw_pts = None

        if do_detect:
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (640, 360), interpolation=cv2.INTER_AREA)
            detections = self.detector.detect(gray_small)
            sx = color.shape[1] / gray_small.shape[1]
            sy = color.shape[0] / gray_small.shape[0]

            if detections:
                tag = detections[0]
                raw_pts = self.reorder_corners_ccw(tag.corners.astype(float))
                raw_pts[:, 0] *= sx
                raw_pts[:, 1] *= sy
                
                pts = raw_pts.reshape(-1, 1, 2)
                undistorted = cv2.undistortPoints(
                    pts, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix
                )
                undistorted_pts = undistorted.reshape(-1, 2)

                center_u = np.mean(raw_pts[:, 0])
                center_v = np.mean(raw_pts[:, 1])

                poly = PolygonStamped()
                poly.header.stamp = stamp
                poly.header.frame_id = "camera_color_optical_frame"
                valid = True
                
                for (u_raw, v_raw), (u, v) in zip(raw_pts, undistorted_pts):

                    sample_u = u_raw + (center_u - u_raw) * 0.15
                    sample_v = v_raw + (center_v - v_raw) * 0.15

                    Z = self.sample_depth(depth, sample_u, sample_v)

                    if Z is None:
                        valid = False
                        break
                        
                    p = Point32()
                    p.x, p.y, p.z = float(u), float(v), float(Z)
                    poly.polygon.points.append(p)
                    
                if valid:
                    self.last_poly = poly
                    self.corners_pub.publish(poly)

                elif self.last_poly is not None:
                    self.last_poly.header.stamp = stamp
                    self.corners_pub.publish(self.last_poly)

        # ---------------- Streaming ----------------
        stream = cv2.resize(color, (640, 480))
        sx, sy = 640 / 1280, 480 / 720

        desired_draw = (self.desired * np.array([sx, sy])).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(stream, [desired_draw], True, (0, 0, 255), 2)

        if undistorted_pts is not None:
            pts_u = np.asarray(undistorted_pts).reshape(4, 2)
            pts_u_draw = (pts_u * [sx, sy]).astype(np.int32)
            cv2.polylines(stream, [pts_u_draw], True, (0, 255, 0), 2)

        if raw_pts is not None:
            pts_r = np.asarray(raw_pts).reshape(4, 2)
            pts_r_draw = (pts_r * [sx, sy]).astype(np.int32)
            cv2.polylines(stream, [pts_r_draw], True, (0, 180, 180), 1)

        # --- Error Overlay ---
        if self.current_err is not None:
            e = np.asarray(self.current_err.data)
            
            x0, y0, dy = 153, 25, 20
            num_elements = len(e)
            if num_elements == 12:
                for i in range(4):
                    ex, ey, ez = e[i*3:(i+1)*3]
                    cv2.putText(stream, f"P{i+1}: ex={ex:+.2f}px  ey={ey:+.2f}px  ez={ez:+.2f}",
                        (x0, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            elif num_elements == 3:
                ex, ey, ez = e
                cv2.putText(stream, f"Center Err: X:{ex:+.2f} Y:{ey:+.2f} Z:{ez:+.2f}",
                            (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            elif num_elements == 2:
                ex, ey = e
                cv2.putText(stream, f"Center Err: X:{ex:+.2f} Y:{ey:+.2f}",
                            (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        
        if self.current_vel is not None:
            v = self.current_vel
            cv2.putText(stream, f"Vx:{v.twist.linear.x:+.4f}", (20,30), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0,128,255), 2)
            cv2.putText(stream, f"Vy:{v.twist.linear.y:+.4f}", (20,50), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0,128,255), 2)
            cv2.putText(stream, f"Vz:{v.twist.linear.z:+.4f}", (20,70), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0,128,255), 2)
            cv2.putText(stream, f"Wx:{v.twist.angular.x:+.4f}", (20,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,255), 2)
            cv2.putText(stream, f"Wy:{v.twist.angular.y:+.4f}", (20,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,255), 2)
            cv2.putText(stream, f"Wz:{v.twist.angular.z:+.4f}", (20,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,255), 2)

        if self.current_rc is not None:
            rc = self.current_rc
            cv2.putText(stream, f"Surge :{rc.channels[4]:+.2f}", (20,150), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (51,255,153), 2)
            cv2.putText(stream, f"Sway  :{rc.channels[5]:+.2f}", (20,170), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (51,255,153), 2)
            cv2.putText(stream, f"Heave :{rc.channels[2]:+.2f}", (20,190), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (51,255,153), 2)
            cv2.putText(stream, f"Yaw   :{rc.channels[3]:+.2f}", (20,210), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (51,255,153), 2)

        # --- Actual Motor PWMs ---
        if self.current_rc_out is not None:
            mx, my = 20, 230 
            for i in range(8):
                if i < len(self.current_rc_out.channels):
                    pwm = self.current_rc_out.channels[i]
                    cv2.putText(stream, f"M{i+1}: {pwm}", (mx, my + (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
        # Push to QGC
        self.video_writer.write(stream)

        # ---------------- Image publish (rate limited) ----------------
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_img_pub >= self.img_pub_period:
            self.last_img_pub = now

            raw = self.bridge.cv2_to_imgmsg(color, encoding="bgr8")
            raw.header.stamp = stamp
            raw.header.frame_id = "camera_link"
            self.raw_pub.publish(raw)

            comp = CompressedImage()
            comp.header = raw.header
            comp.format = "jpeg"
            comp.data = cv2.imencode(".jpg", color,
                                     [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            self.comp_pub.publish(comp)

        # ---------------- Depth publish (optional) ----------------
        if self.publish_full_depth:
            depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="16UC1")
            depth_msg.header.stamp = stamp
            self.depth_pub.publish(depth_msg)

    def destroy_node(self):
        self.video_writer.release()
        self.pipeline.stop()
        super().destroy_node()


def main():
    rclpy.init()
    node = IBVS_Telemetry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
