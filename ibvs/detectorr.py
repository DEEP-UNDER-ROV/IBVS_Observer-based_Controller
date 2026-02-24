#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np

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

    def reorder_corners_ccw(self, pts):
        center = np.mean(pts, axis=0)
    
        ordered = []
    
        for p in pts:
            if p[0] < center[0] and p[1] < center[1]:
                ordered.append(("TL", p))
            elif p[0] > center[0] and p[1] < center[1]:
                ordered.append(("TR", p))
            elif p[0] > center[0] and p[1] > center[1]:
                ordered.append(("BR", p))
            else:
                ordered.append(("BL", p))
    
        ordered_dict = {name: point for name, point in ordered}
    
        return np.array([
            ordered_dict["TL"],
            ordered_dict["TR"],
            ordered_dict["BR"],
            ordered_dict["BL"]
        ], dtype=float)

    def __init__(self):
        from message_filters import Subscriber, ApproximateTimeSynchronizer
        super().__init__("IBVS_Telemetry")
        self.get_logger().info("Telemetry Node Started")
        self.bridge = CvBridge()

        # ---------------- Publishers ----------------
        self.corners_pub = self.create_publisher(PolygonStamped, "/apriltag/corners", 10)

        # ---------------- Subscriptions ----------------
        self.color_sub = Subscriber(self, Image, "/camera/color/image_raw")
        self.depth_sub = Subscriber(self, Image, "/camera/depth/image_rect_raw")
        self.ts = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.sync_callback)
        
        self.create_subscription(OverrideRCIn, "/mavros/rc/override", self.cb_rc, 10)
        self.create_subscription(RCOut, "/mavros/rc/out", self.cb_rc_out, 10)
        self.create_subscription(Twist, "/ibvs/vel", self.cb_vel, 10)
        self.create_subscription(Float32MultiArray, "/ibvs/error", self.cb_err, 10)

        self.current_rc = None
        self.current_rc_out = None
        self.current_vel = None
        self.current_err = None

        self.camera_matrix = np.array([[FX, 0, CX],
                                       [0, FY, CY],
                                       [0,  0,  1]], dtype=np.float32)
        self.dist_coeffs = np.array(DIST_COEFFS, dtype=np.float32)
        self.desired = self.desired_corners_from_Z(Z_DES)

        # ---------------- AprilTag ----------------
        self.detector = Detector(
            families="tag36h11",
            nthreads=2,
            quad_decimate=0.7,
            quad_sigma=0.8,
            refine_edges=True
        )

        # ---------------- Efficiency controls ----------------
        self.frame_count = 0
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

        self.get_logger().info("IBVS Telemetry running")
        self.get_logger().info(f"Streaming to QGC at {QGC_IP}:{QGC_PORT}")

    # ---------------- Callbacks ----------------
    def cb_rc(self, msg): self.current_rc = msg
    def cb_rc_out(self, msg): self.current_rc_out = msg
    def cb_vel(self, msg): self.current_vel = msg
    def cb_err(self, msg): self.current_err = msg

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

    def sync_callback(self, color_msg, depth_msg):

        color = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg)

        stamp = color_msg.header.stamp
        self.frame_count += 1

        raw_pts = None
        undistorted_pts = None

        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (640, 360), interpolation=cv2.INTER_AREA)
        detections = self.detector.detect(gray_small)
        sx = color.shape[1] / gray_small.shape[1]
        sy = color.shape[0] / gray_small.shape[0]

        if detections:
            tag = detections[0]
            raw_pts = self.reorder_corners_ccw(tag.corners.astype(float))

            pts = raw_pts.reshape(-1, 1, 2)
            undistorted = cv2.undistortPoints(
                pts, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
            undistorted_pts = undistorted.reshape(-1, 2)

            poly = PolygonStamped()
            poly.header.stamp = stamp
            poly.header.frame_id = "camera_color_optical_frame"

            valid = True
            for (u, v) in raw_pts:
                Z = self.sample_depth(depth, u, v)
                if Z is None:
                    valid = False
                    break

                p = Point32()
                p.x, p.y, p.z = float(u), float(v), float(Z)
                poly.polygon.points.append(p)

            if valid:
                self.corners_pub.publish(poly)

        self.process_overlay_and_stream(color, raw_pts, undistorted_pts)

    def process_overlay_and_stream(self, color, raw_pts, undistorted_pts):
        stream = cv2.resize(color, (640, 480))
        h, w = color.shape[:2]
        sx, sy = 640 / w, 480 / h

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
                ex, ey = e
                cv2.putText(stream, f"Center Err: X:{ex:+.2f} Y:{ey:+.2f}",
                            (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        
        if self.current_vel is not None:
            v = self.current_vel
            cv2.putText(stream, f"Vx:{v.linear.x:+.2f}", (20,45), 2,  0.5, (0,128,255), 2)
            cv2.putText(stream, f"Vy:{v.linear.y:+.2f}", (20,65), 2,  0.5, (0,128,255), 2)
            cv2.putText(stream, f"Vz:{v.linear.z:+.2f}", (20,85), 2,  0.5, (0,128,255), 2)
            cv2.putText(stream, f"Wx:{v.angular.x:+.2f}", (20,105), 2, 0.5, (0,128,255), 2)
            cv2.putText(stream, f"Wy:{v.angular.y:+.2f}", (20,125), 2, 0.5, (0,128,255), 2)
            cv2.putText(stream, f"Wz:{v.angular.z:+.2f}", (20,145), 2, 0.5, (0,128,255), 2)

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
