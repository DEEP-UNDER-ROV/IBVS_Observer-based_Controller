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

from ibvs.parameter import *


class VideoStreamer(Node):
    def __init__(self):
        super().__init__("Video_Streamer")
        self.declare_parameter("mode", "auto")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("overlay_image_topic", "/corrected/left/image_raw")

        self.mode = self.get_parameter("mode").value
        self.image_topic = self.get_parameter("image_topic").value
        self.overlay_image_topic = self.get_parameter("overlay_image_topic").value

        self.get_logger().info(f"Streaming Mode: {self.mode}")

        self.bridge = CvBridge()

        self.active_mode = None

        self.detected_corners = None
        self.current_rc = None
        self.current_rc_out = None
        self.current_vel = None
        self.current_err = None
        self.last_poly = None

        self.desired, R = self.compute_desired_corners_pixel(Z_DES=Z_DES, pitch_deg=PITCH_DES_DEG, yaw_deg=YAW_DES_DEG, roll_deg=ROLL_DES_DEG)

        # ---------------- Publishers ----------------
        # overlay image publishers (overlay -> ROS + compressed for QGC)
        self.overlay_pub = self.create_publisher(Image, "/camera/overlay/image_raw", 10)
        self.comp_pub    = self.create_publisher(CompressedImage, "/camera/overlay/image_raw/compressed", 10)

        self.video_writer = None

        if self.mode == "auto":
            self.get_logger().info("Waiting for ROS topic discovery...")
            self.auto_timer = self.create_timer(1.0, self.auto_detect_mode)

        else:
            self.configure_mode(self.mode)

        self.img_pub_period = 1.0 / 20.0 # image @ 20 Hz
        self.last_img_pub = 0.0

    # =========================================================
    def auto_detect_mode(self):
        topics = dict(self.get_topic_names_and_types())
        detection_exists = "/detection1" in topics

        if detection_exists:
            self.get_logger().info("Detected /detection1 -> switching to OVERLAY mode")
            self.configure_mode("overlay")

        else:
            self.get_logger().info("No /detection1 detected -> switching to CLEAN mode")
            self.configure_mode("clean")

        self.destroy_timer(self.auto_timer)
        self.auto_timer = None

    # =========================================================
    def configure_mode(self, mode):
        if mode not in ["overlay", "clean"]:
            self.get_logger().error(f"Invalid mode: {mode}")
            return

        self.active_mode = mode

        gst_pipeline = (
            f"appsrc is-live=true block=true do-timestamp=true format=time ! "
            f"queue ! "
            f"videoconvert ! "
            f"video/x-raw,width={stream_w},height={stream_h},format=I420 ! "
            f"x264enc tune=zerolatency "
            f"bitrate=2000 speed-preset=ultrafast "
            f"key-int-max=30 ! "
            f"rtph264pay config-interval=-1 pt=96 ! "
            f"udpsink host={QGC_IP} port={QGC_PORT} sync=false async=false"
        )

        self.video_writer = cv2.VideoWriter(
            gst_pipeline, cv2.CAP_GSTREAMER, 0, 30, (stream_w, stream_h),True)

        if not self.video_writer.isOpened():
            self.get_logger().error("GStreamer pipeline failed!")

        self.get_logger().info(f"Streaming {mode.upper()} video to {QGC_IP}:{QGC_PORT}")

        if mode == "overlay":
            self.configure_overlay_subscriptions()

        else:
            self.configure_clean_subscriptions()

    # =========================================================
    def configure_clean_subscriptions(self):
        self.get_logger().info(f"CLEAN mode image topic: {self.image_topic}")
        self.create_subscription(Image, self.image_topic, self.cb_image_clean, 10)

    # =========================================================
    def configure_overlay_subscriptions(self):
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        # ---------------- Subscriptions ----------------
        self.create_subscription(AprilTagDetectionArray, "/detection1", self.cb_detection, 10)
        self.create_subscription(OverrideRCIn, "/mavros/rc/override", self.cb_rc, 10)
        self.create_subscription(TwistStamped, "/ibvs/nu_B_hat", self.cb_vel_hat, 10)
        self.create_subscription(Float32MultiArray, "/ibvs/error/px", self.cb_err, 10)

        self.create_subscription(PolygonStamped,"/apriltag/corners",self.cb_corners,qos)
        self.create_subscription(Image, "/corrected/left/image_raw", self.cb_image_overlay, 10)

        self.get_logger().info(
            f"OVERLAY mode image topic: "
            f"{self.overlay_image_topic}"
        )

        self.detected_corners = None
        self.current_rc = None
        self.current_rc_out = None
        self.current_vel_hat = None
        self.current_err = None
        self.last_poly = None      
       

    # ---------------- Callbacks ----------------
    def cb_rc(self, msg): self.current_rc = msg
    def cb_vel_hat(self, msg): self.current_vel_hat = msg
    def cb_err(self, msg): self.current_err = msg
    def cb_corners(self, msg): self.last_poly = msg

    # =========================================================
    def cb_detection(self, msg):
        if not msg.detections:
            self.detected_corners = None
            return
    
        det = msg.detections[0]
        self.detected_corners = np.array([[c.x, c.y] for c in det.corners],dtype=np.float32)

    # =========================================================
    def cb_image_clean(self, msg):
        try:
            if "infra" in self.image_topic.lower():
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            else:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        except Exception as e:
            self.get_logger().warn(f"Failed to convert clean frame: {e}")
            return

        frame = self.resize_for_stream(frame)
        self.push_frame(frame, msg.header.stamp, publish_overlay=False)

    # =========================================================
    def cb_image_overlay(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        except Exception as e:
            self.get_logger().warn(f"Failed to convert overlay frame: {e}")
            return

        stream = frame.copy()

        desired_draw = (self.desired .astype(np.int32) .reshape(-1, 1, 2))
        cv2.polylines(stream, [desired_draw], True, (0, 0, 255),2)

        if self.detected_corners is not None:
            pts = (self.detected_corners.reshape((-1, 1, 2)).astype(np.int32))
            cv2.polylines(stream, [pts], True, (0, 255, 0), 2)

        self.draw_error(stream)
        self.draw_rc(stream)

        stream = self.resize_for_stream(stream)

        self.push_frame(stream,
            msg.header.stamp,
            publish_overlay=True,
            frame_id=msg.header.frame_id)

    # =========================================================
    def compute_desired_corners_pixel(self, Z_DES, pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0):
        half = TAG_SIZE / 2.0

        # Tag corners in tag frame (meters)
        corners = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0]
        ], dtype=float)

        # Desired roll (rotation about image/camera Z-axis)
        rx = np.deg2rad(pitch_deg)
        ry = np.deg2rad(yaw_deg)
        rz = np.deg2rad(roll_deg)

        Rx = np.array([
            [1,0,0],
            [0,np.cos(rx),-np.sin(rx)],
            [0,np.sin(rx), np.cos(rx)]
        ])

        Ry = np.array([
            [ np.cos(ry),0,np.sin(ry)],
            [0,1,0],
            [-np.sin(ry),0,np.cos(ry)]
        ])

        Rz = np.array([
            [np.cos(rz),-np.sin(rz),0],
            [np.sin(rz), np.cos(rz),0],
            [0,0,1]
        ])

        # Rotate corners
        R = Rz @ Ry @ Rx
        corners = (R @ corners.T).T
        corners[:,2] += Z_DES

        # Perspective projection
        x = corners[:, 0] / corners[:,2]
        y = corners[:, 1] / corners[:,2]

        # Convert to pixels
        u = FX * x + CX
        v = FY * y + CY

        desired_pixels = np.column_stack((u, v))

        return desired_pixels, R

    # =========================================================
    def draw_error(self, stream):
        if self.current_err is None:
            return

        e = np.asarray(self.current_err.data,dtype=float)

        x0 = 175
        y0 = 20
        dy = 20

        num_elements = len(e)
        if num_elements == 12:
            for i in range(4):
                eu, ev, eZ = e[i * 3:(i + 1) * 3]
                cv2.putText(stream, f"P{i+1}: eu={eu:+.2f}px ev={ev:+.2f}px eZ={eZ:+.2f}m",
                    (x0, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        elif num_elements == 8:
            e = e.reshape(4, 2)
            for i in range(4):
                eu, ev = e[i]
                cv2.putText(stream, f"P{i+1}: eu={eu:+.2f} ev={ev:+.2f}",
                    (x0, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        else:
            cv2.putText(
                stream,f"ERR: {num_elements} values", (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # =========================================================
    def draw_rc(self, stream):
        if self.current_rc is None:
            return
        
        rc = self.current_rc
        try:
            cv2.putText(stream, f"Surge :{rc.channels[4]:+.2f}", (20,150), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (51,255,153), 2)
            cv2.putText(stream, f"Sway  :{rc.channels[5]:+.2f}", (20,170), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (51,255,153), 2)
            cv2.putText(stream, f"Heave :{rc.channels[2]:+.2f}", (20,190), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (51,255,153), 2)
            cv2.putText(stream, f"Roll  :{rc.channels[1]:+.2f}", (20,250), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (51,255,153), 2)
            cv2.putText(stream, f"Pitch :{rc.channels[0]:+.2f}", (20,230), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (51,255,153), 2)
            cv2.putText(stream, f"Yaw   :{rc.channels[3]:+.2f}", (20,210), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (51,255,153), 2)

        except IndexError:
            pass
            
    # =========================================================
    def draw_nu_hat(self, stream):
        if self.current_rc is None:
            return
        
        rc = self.current_rc
        try:
            cv2.putText(stream, f"Surge :{rc.channels[4]:+.2f}", (20,150), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (51,255,153), 2)
            cv2.putText(stream, f"Sway  :{rc.channels[5]:+.2f}", (20,170), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (51,255,153), 2)
            cv2.putText(stream, f"Heave :{rc.channels[2]:+.2f}", (20,190), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (51,255,153), 2)
            cv2.putText(stream, f"Roll  :{rc.channels[1]:+.2f}", (20,250), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (51,255,153), 2)
            cv2.putText(stream, f"Pitch :{rc.channels[0]:+.2f}", (20,230), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (51,255,153), 2)
            cv2.putText(stream, f"Yaw   :{rc.channels[3]:+.2f}", (20,210), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (51,255,153), 2)

        except IndexError:
            pass
            
    # =========================================================
    def resize_for_stream(self, frame):
        h, w = frame.shape[:2]
        if (h, w) != (stream_h, stream_w):

            frame = cv2.resize(frame,
                (stream_w, stream_h),
                interpolation=cv2.INTER_LINEAR)

        return frame

    # =========================================================
    def push_frame(self, frame, stamp, 
        publish_overlay=False,
        frame_id="camera_link"):

        if self.video_writer is not None:
            if self.video_writer.isOpened():
                self.video_writer.write(frame)

        # Only publish ROS overlay topics in overlay mode
        if not publish_overlay:
            return

        now = (self.get_clock().now().nanoseconds* 1e-9)
        if now - self.last_img_pub < self.img_pub_period:
            return

        self.last_img_pub = now

        overlay = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        overlay.header.stamp = stamp
        overlay.header.frame_id = (frame_id
            if frame_id
            else "camera_link")

        self.overlay_pub.publish(overlay)

        encoded = cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,80])

        if encoded[0]:
            comp = CompressedImage()
            comp.header = overlay.header
            comp.format = "jpeg"
            comp.data = encoded[1].tobytes()
            self.comp_pub.publish(comp)

    # =========================================================
    def destroy_node(self):
        if self.video_writer is not None:
            if self.video_writer.isOpened():
                self.video_writer.release()
                self.get_logger().info("GStreamer pipeline released.")

        super().destroy_node()


def main():
    rclpy.init()
    node = VideoStreamer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
