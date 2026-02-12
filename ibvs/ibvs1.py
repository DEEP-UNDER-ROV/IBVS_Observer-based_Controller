#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import numpy as np

from geometry_msgs.msg import PolygonStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int16MultiArray
from mavros_msgs.msg import OverrideRCIn
from cv_bridge import CvBridge

from ibvs.constants import *

class IBVSRCController(Node):
    def __init__(self):
        super().__init__("ibvs_rc_controller")

        self.bridge = CvBridge()

        # ---------------- Subscribers ----------------
        self.sub_corners = self.create_subscription(
            PolygonStamped,
            "/apriltag/corners",
            self.cb_corners,
            qos_profile_sensor_data
        )

        self.sub_depth = self.create_subscription(
            Image,
            "/camera/depth/image_raw",
            self.cb_depth,
            qos_profile_sensor_data
        )

        # ---------------- Publishers ----------------
        self.pwm_pub = self.create_publisher(
            Int16MultiArray,
            "/ibvs/pwm_debug",
            10
        )

        self.vel_pub = self.create_publisher(
            Float32MultiArray,
            "/ibvs/velocity_debug",
            10
        )

        self.err_pub = self.create_publisher(
            Float32MultiArray,
            "/ibvs/error",
            10
        )

        # ---------------- State ----------------
        self.depth_img = None
        self.last_tag_time = None
        self.tag_lost = True
        self.TAG_TIMEOUT = 0.5  # seconds

        # RC command buffer (IMPORTANT)
        self.rc_cmd = [1500] * 18

        # ---------------- Gains (TUNE IN WATER) ----------------
        self.K_SURGE = 300
        self.K_SWAY  = 300
        self.K_HEAVE = 350
        self.K_YAW   = 220
        self.HEAVE_BIAS = 40

        # ---------------- Desired image features ----------------
        self.desired_pts = self.compute_desired_corners(
            Z_DES, FX, FY, CX, CY, TAG_SIZE
        )

        # ---------------- Timers ----------------
        self.create_timer(0.1, self.tag_watchdog)

        self.get_logger().info("IBVS RC-Override Controller READY (EKF BYPASSED)")

    # =========================================================
    def compute_desired_corners(self, Z_des, fx, fy, cx, cy, tag_size):
        s = tag_size / 2.0
        corners = np.array([
            [-s, -s, Z_des],
            [ s, -s, Z_des],
            [ s,  s, Z_des],
            [-s,  s, Z_des],
        ])

        pts = np.zeros((4, 2))
        for i, (X, Y, Z) in enumerate(corners):
            pts[i, 0] = fx * X / Z + cx
            pts[i, 1] = fy * Y / Z + cy
        return pts

    # =========================================================
    def cb_depth(self, msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001

    # =========================================================
    def interaction_matrix(self, u, v, Z):
        x = (u - CX) / FX
        y = (v - CY) / FY
        return np.array([
            [-1/Z,  0,    x/Z,  x*y,      -(1 + x*x),  y],
            [0,    -1/Z,  y/Z,  1 + y*y,  -x*y,       -x],
            [0,     0,   -1/Z,  -y,        x,          0]
        ])

    # =========================================================
    def vel_to_pwm(self, v, gain, bias=0):
        return int(np.clip(1500 + gain * v + bias, 1100, 1900))

    # =========================================================
    def cb_corners(self, msg):
        if len(msg.polygon.points) != 4:
            return

        self.last_tag_time = self.get_clock().now()
        self.tag_lost = False

        rows = []
        errs = []
        pixel_err = []
        h, w = self.depth_img.shape

        for i, p in enumerate(msg.polygon.points):
            u, v, Z = p.x, p.y, p.z
            if Z <= 0:
                return
                
            ud, vd = self.desired_pts[i]
            pixel_err.append(u - ud)
            pixel_err.append(v - vd)
            rows.append(self.interaction_matrix(u, v, Z))

            x, y = (u - CX)/FX, (v - CY)/FY
            xd, yd = (self.desired_pts[i] - [CX, CY]) / [FX, FY]
            errs.extend([x - xd, y - yd, Z - Z_DES])
            # errs.extend([x - xd, y - yd, 0.25*(Z - Z_DES)])

        err_array = np.array(errs).reshape(4, 3)
        L = np.vstack(rows)
        e = np.array(errs).reshape(-1, 1)

        mu = 1
        A = L.T @ L + mu**2 * np.eye(6)
        b = L.T @ e
        Vc = -LAMBDA_P * np.linalg.solve(A, b)

        if np.mean(np.abs(pixel_err)) < 10.0:
            Vc[:] = 0.0

        v_c = Vc[0:3]
        w_c = Vc[3:6]

        v_c = v_c.reshape(3,)
        w_c = w_c.reshape(3,)
        
        Wb = (R_CB @ w_c).reshape(3,)
        Vb = (R_CB @ v_c).reshape(3,) + np.cross(Wb, P_CB.reshape(3,))

        vel_msg = Float32MultiArray()
        vel_msg.data = Vb.flatten().tolist() + Wb.flatten().tolist()
        self.vel_pub.publish(vel_msg)

        # -------- Compute PWM --------
        pwm = [1500] * 18
        
        pwm[4] = self.vel_to_pwm(Vb[0], self.K_SURGE)
        pwm[5] = self.vel_to_pwm(Vb[1], self.K_SWAY)
        pwm[2] = self.vel_to_pwm(Vb[2], self.K_HEAVE, self.HEAVE_BIAS)
        pwm[3] = self.vel_to_pwm(Wb[2], self.K_YAW)
        
        # Publish debug PWM
        msg_pwm = Int16MultiArray()
        msg_pwm.data = pwm
        self.pwm_pub.publish(msg_pwm)

        self.get_logger().info(
            f"PWM CMD | surge={self.rc_cmd[4]} "
            f"sway={self.rc_cmd[5]} "
            f"heave={self.rc_cmd[2]} "
            f"yaw={self.rc_cmd[3]}",
            throttle_duration_sec=0.5
        )

        err_msg = Float32MultiArray()
        err_msg.data = np.array(errs, dtype=np.float32).tolist()
        self.err_pub.publish(err_msg)

    # =========================================================
    def tag_watchdog(self):
        if self.last_tag_time is None:
            return

        dt = (self.get_clock().now() - self.last_tag_time).nanoseconds * 1e-9
        if dt > self.TAG_TIMEOUT and not self.tag_lost:
            self.tag_lost = True
            neutral = Int16MultiArray()
            neutral.data = [1500] * 18
            self.pwm_pub.publish(neutral)
            self.get_logger().warn("AprilTag LOST → RC neutral")


# =============================================================
def main():
    rclpy.init()
    rclpy.spin(IBVSRCController())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
