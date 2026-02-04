#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import pyrealsense2 as rs

from cv_bridge import CvBridge
from pupil_apriltags import Detector

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PolygonStamped, Point32, Twist, Point, Vector3Stamped
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
        self.create_subscription(Point, "/ibvs/pos", self.cb_pos, 10)
        self.create_subscription(Twist, "/ibvs/vel", self.cb_vel, 10)
        self.create_subscription(Float32MultiArray, "/ibvs/error", self.cb_err, 10)
        self.create_subscription(Vector3Stamped, "/pnp/tvec", self.cb_tvec, 10)

        self.current_rc = None
        self.current_rc_out = None
        self.current_vel = None
        self.current_pos = None
        self.current_tvec = None
        self.current_err = None

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

        self.bridge = CvBridge()

        # ---------------- AprilTag ----------------
        self.detector = Detector(
            families="tag36h11",
            nthreads=2,
            quad_decimate=1.0,
            refine_edges=True
        )

        # ---------------- Efficiency controls ----------------
        self.detect_div = 3              # 30 Hz → 10 Hz detection
        self.frame_count = 0

        self.publish_full_depth = False  # enable only for rosbag
        self.img_pub_period = 1.0 / 20.0 # image @ 20 Hz
        self.last_img_pub = 0.0

        self.PATCH = 3

        # ---------------- Video Stream ----------------
        gst_pipeline = (
            f"appsrc ! videoconvert ! "
            f"video/x-raw,width=640,height=480,format=I420 ! "
            f"x264enc tune=zerolatency bitrate=1000 speed-preset=ultrafast ! "
            f"rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={QGC_IP} port={QGC_PORT} sync=false"
        )
        self.video_writer = cv2.VideoWriter(
            gst_pipeline, cv2.CAP_GSTREAMER, 0, 30, (640, 480), True
        )

        self.timer = self.create_timer(1.0 / 30.0, self.loop)
        self.get_logger().info("IBVS Telemetry running")

    # ---------------- Callbacks ----------------
    def cb_rc(self, msg): self.current_rc = msg
    def cb_rc_out(self, msg): self.current_rc_out = msg
    def cb_vel(self, msg): self.current_vel = msg
    def cb_pos(self, msg): self.current_pos = msg
    def cb_tvec(self, msg): self.current_tvec = msg
    def cb_err(self, msg): self.current_err = msg

    @staticmethod
    def order_corners_apriltag(c):
        return np.array([c[3], c[2], c[1], c[0]], dtype=np.float32)

    def sample_depth(self, depth, u, v):
        h, w = depth.shape
        ui, vi = int(u), int(v)

        if not (0 <= ui < w and 0 <= vi < h):
            return None

        patch = depth[
            max(0, vi-self.PATCH):min(h, vi+self.PATCH+1),
            max(0, ui-self.PATCH):min(w, ui+self.PATCH+1)
        ]

        valid = patch[patch > 0]
        if valid.size == 0:
            return None

        return float(np.median(valid)) * 0.001  # mm → m

    # ---------------- Main loop ----------------
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
        stamp = self.get_clock().now().to_msg()

        self.frame_count += 1
        do_detect = (self.frame_count % self.detect_div) == 0

        undistorted_pts = None
        raw_pts = None

        if do_detect:
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (640, 360), interpolation=cv2.INTER_AREA)
            detections = self.detector.detect(gray_small)

            if detections:
                tag = detections[0]
                raw_pts = self.order_corners_apriltag(tag.corners)
                raw_pts[:, 0] *= 2.0
                raw_pts[:, 1] *= 2.0

                pts = raw_pts.reshape(-1, 1, 2)
                undistorted = cv2.undistortPoints(
                    pts, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix
                )
                undistorted_pts = undistorted.reshape(-1, 2)

                poly = PolygonStamped()
                poly.header.stamp = stamp
                poly.header.frame_id = "camera_color_optical_frame"
                
                for (u, v) in undistorted_pts:
                    Z = self.sample_depth(depth, u, v)
                    if Z is None:
                        return
                        
                    p = Point32()
                    p.x, p.y, p.z = float(u), float(v), Z
                    poly.polygon.points.append(p)
                self.corners_pub.publish(poly)

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
                    cv2.putText(stream, f"P{i+1}: ex={ex:+.2f}px  ey={ey:+.2f}px  ez={ez:+.2f} m",
                        (x0, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            elif num_elements == 3:
                ex, ey, ez = e
                cv2.putText(stream, f"Center Err: X:{ex:+.2f} Y:{ey:+.2f} Z:{ez:+.2f}",
                            (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            elif num_elements == 2:
                ex, ey = e[i*2:(i+1)*2]
                cv2.putText(stream, f"Center Err: X:{ex:+.2f} Y:{ey:+.2f}",
                            (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        
        if self.current_vel is not None:
            v = self.current_vel
            cv2.putText(stream, f"Vx:{v.linear.x:+.2f}", (20,45), 2,  0.5, (0,128,255), 2)
            cv2.putText(stream, f"Vy:{v.linear.y:+.2f}", (20,65), 2,  0.5, (0,128,255), 2)
            cv2.putText(stream, f"Vz:{v.linear.z:+.2f}", (20,85), 2,  0.5, (0,128,255), 2)
            cv2.putText(stream, f"Wx:{v.angular.x:+.2f}", (20,105), 2, 0.5, (0,128,255), 2)
            cv2.putText(stream, f"Wy:{v.angular.y:+.2f}", (20,125), 2, 0.5, (0,128,255), 2)
            cv2.putText(stream, f"Wz:{v.angular.z:+.2f}", (20,125), 2, 0.5, (0,128,255), 2)

        if self.current_tvec is not None:
            z_val = self.current_tvec.vector.z
            cv2.putText(stream, f"Range:{z_val:.3f} m", (20,25), 2,  0.5, (51,255,153), 2)
            
        if self.current_pos is not None:
            p = self.current_pos
            cv2.putText(stream, f"Offset X:{p.x:+.2f}m", (20,150), 2,  0.5, (51,255,153), 2)
            cv2.putText(stream, f"Offset Y:{p.y:+.2f}m", (20,170), 2,  0.5, (51,255,153), 2)
            cv2.putText(stream, f"Offset Z:{p.z:+.2f}m", (20,190), 2,  0.5, (51,255,153), 2)

        if self.current_rc is not None:
            rc = self.current_rc
            cv2.putText(stream, f"Surge :{rc.channels[4]:+.2f}", (20,150), 2,  0.5, (51,255,153), 2)
            cv2.putText(stream, f"Sway  :{rc.channels[5]:+.2f}", (20,170), 2,  0.5, (51,255,153), 2)
            cv2.putText(stream, f"Heave :{rc.channels[2]:+.2f}", (20,190), 2,  0.5, (51,255,153), 2)
            cv2.putText(stream, f"Yaw   :{rc.channels[3]:+.2f}", (20,210), 2,  0.5, (51,255,153), 2)

        # --- Actual Motor PWMs ---
        if self.current_rc_out is not None:
            mx, my = 20, 230 
            for i in range(8):
                if i < len(self.current_rc_out.channels):
                    pwm = self.current_rc_out.channels[i]
                    cv2.putText(stream, f"M{i+1}: {pwm}", (mx, my + (i * 20)), 2, 0.5, (0, 255, 0), 1)
                    
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
