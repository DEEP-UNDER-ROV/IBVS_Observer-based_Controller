#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PolygonStamped, Point32, Twist, Point, Vector3Stamped, TwistStamped
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from mavros_msgs.msg import OverrideRCIn, RCOut
from apriltag_msgs.msg import AprilTagDetectionArray

from ibvs.constants import *


class IBVS_Telemetry(Node):
    def compute_desired_corners(self, Z_des):
        half = TAG_SIZE / 2.0
    
        corners_3d = np.array([
            [-half, -half, Z_des],
            [ half, -half, Z_des],
            [ half,  half, Z_des],
            [-half,  half, Z_des],
        ])
    
        desired = np.zeros((4,2), dtype=np.float32)
    
        for i,(X,Y,Z) in enumerate(corners_3d):
            desired[i] = [X/Z, Y/Z]   # normalized coordinates
    
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
        # overlay image publishers (overlay -> ROS + compressed for QGC)
        self.overlay_pub = self.create_publisher(Image, "/camera/overlay/image_raw", 10)
        self.comp_pub    = self.create_publisher(CompressedImage, "/camera/overlay/image_raw/compressed", 10)

        # ---------------- Subscriptions ----------------
        self.create_subscription(AprilTagDetectionArray, "/detection1", self.cb_detection, 10)
        self.create_subscription(OverrideRCIn, "/mavros/rc/override", self.cb_rc, 10)
        self.create_subscription(RCOut, "/mavros/rc/out", self.cb_rc_out, 10)
        self.create_subscription(TwistStamped, "/ibvs/vel", self.cb_vel, 10)
        self.create_subscription(Float32MultiArray, "/ibvs/error", self.cb_err, 10)
        
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.create_subscription(PolygonStamped,"/apriltag/corners",self.cb_corners,qos)
        self.create_subscription(Image, "/corrected/left/image_raw", self.cb_image, 10)

        self.detected_corners = None
        self.current_rc = None
        self.current_rc_out = None
        self.current_vel = None
        self.current_err = None
        self.last_poly = None

        self.camera_matrix = np.array([[FX, 0, CX],
                                       [0, FY, CY],
                                       [0,  0,  1]], dtype=np.float32)
        self.dist_coeffs = np.array(DIST_COEFFS, dtype=np.float32)
        self.desired = self.compute_desired_corners(Z_DES)

        
        self.bridge = CvBridge()

        # publishing rate control (for overlay publish)
        self.img_pub_period = 1.0 / 20.0 # image @ 20 Hz
        self.last_img_pub = 0.0

        self.PATCH = PATCH
        
        gst_pipeline = (
            f"appsrc is-live=true block=true do-timestamp=true format=time ! "
            f"queue ! "
            f"videoconvert ! "
            f"video/x-raw,width=848,height=480,format=I420 ! "
            f"x264enc tune=zerolatency "
            f"bitrate=2000 speed-preset=ultrafast "
            f"key-int-max=30 ! "
            f"rtph264pay config-interval=-1 pt=96 ! "
            f"udpsink host={QGC_IP} port={QGC_PORT} sync=false async=false"
        )
        
        self.video_writer = cv2.VideoWriter(
            gst_pipeline, cv2.CAP_GSTREAMER, 0, 30, (848, 480), True
        )

        if not self.video_writer.isOpened():
            self.get_logger().error("GStreamer pipeline failed!")

        self.get_logger().info("IBVS Telemetry running (subscribed image mode)")

    # ---------------- Callbacks ----------------
    def cb_rc(self, msg): self.current_rc = msg
    def cb_rc_out(self, msg): self.current_rc_out = msg
    def cb_vel(self, msg): self.current_vel = msg
    def cb_err(self, msg): self.current_err = msg

    def cb_detection(self, msg):
        if not msg.detections:
            self.detected_corners = None
            return
    
        det = msg.detections[0]
    
        pts = np.array([[c.x, c.y] for c in det.corners], dtype=np.float32)
        self.detected_corners = self.reorder_corners_ccw(pts)
    
    # ---------------- Callbacks for subscribed image + corners ----------------
    def cb_corners(self, msg):
        # store latest corners polygon (assumed to contain Point32 entries with pixel u,v in x,y)
        self.last_poly = msg

    def cb_image(self, msg):
        try:
            color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return

        stamp = msg.header.stamp

        undistorted_pts = None
        raw_pts = None

        # if apriltag corners available, extract and undistort
        if self.last_poly is not None and len(self.last_poly.polygon.points) >= 4:
            pts = self.last_poly.polygon.points
            raw_pts = np.array([[p.x, p.y] for p in pts], dtype=float)
            pts_cv = raw_pts.reshape(-1, 1, 2)
            # und = cv2.undistortPoints(pts_cv, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
            # undistorted_pts = und.reshape(-1, 2)
            undistorted_pts = raw_pts

        # prepare stream for overlay
        h, w = color.shape[:2]
        stream = color.copy()
        sx, sy = 1.0, 1.0

        desired_pixels = np.zeros_like(self.desired)

        for i,(xd,yd) in enumerate(self.desired):
            u = FX * xd + CX
            v = FY * yd + CY
            desired_pixels[i] = [u,v]
        
        desired_draw = desired_pixels.astype(np.int32).reshape(-1,1,2)
        cv2.polylines(stream, [desired_draw], True, (0, 0, 255), 2)

        if self.detected_corners is not None:
            pts = self.detected_corners.reshape((-1,1,2)).astype(np.int32)
            cv2.polylines(stream,[pts],True,(0,255,0),2)

        # if undistorted_pts is not None:
        #     pts_u = np.asarray(undistorted_pts).reshape(-1, 2)
        #     pts_u_draw = (pts_u * [sx, sy]).astype(np.int32)
        #     cv2.polylines(stream, [pts_u_draw], True, (0, 255, 0), 2)

        if raw_pts is not None:
            pts_r = np.asarray(raw_pts).reshape(-1, 2)
            pts_r_draw = (pts_r * [sx, sy]).astype(np.int32)
            cv2.polylines(stream, [pts_r_draw], True, (0, 180, 180), 1)

        # --- Error Overlay ---
        if self.current_err is not None:
            e = np.asarray(self.current_err.data)
            x0, y0, dy = 175, 20, 20
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

        # RC inputs
        if self.current_rc is not None:
            rc = self.current_rc
            cv2.putText(stream, f"Surge :{rc.channels[4]:+.2f}", (20,150), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (51,255,153), 2)
            cv2.putText(stream, f"Sway  :{rc.channels[5]:+.2f}", (20,170), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (51,255,153), 2)
            cv2.putText(stream, f"Heave :{rc.channels[2]:+.2f}", (20,190), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (51,255,153), 2)
            cv2.putText(stream, f"Yaw   :{rc.channels[3]:+.2f}", (20,210), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (51,255,153), 2)
            cv2.putText(stream, f"Pitch :{rc.channels[1]:+.2f}", (20,230), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (51,255,153), 2)
            cv2.putText(stream, f"Roll  :{rc.channels[6]:+.2f}", (20,250), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (51,255,153), 2)

        # Push to QGC
        self.video_writer.write(stream)

        # rate-limited publish of overlay
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_img_pub >= self.img_pub_period:
            self.last_img_pub = now

            overlay = self.bridge.cv2_to_imgmsg(stream, encoding="bgr8")
            overlay.header.stamp = stamp
            overlay.header.frame_id = msg.header.frame_id if msg.header.frame_id else "camera_link"
            self.overlay_pub.publish(overlay)

            comp = CompressedImage()
            comp.header = overlay.header
            comp.format = "jpeg"
            comp.data = cv2.imencode(".jpg", stream, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            self.comp_pub.publish(comp)

    def destroy_node(self):
        self.video_writer.release()
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
