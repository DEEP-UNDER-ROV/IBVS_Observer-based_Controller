#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import numpy as np

from geometry_msgs.msg import PolygonStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from mavros_msgs.msg import OverrideRCIn
from cv_bridge import CvBridge

from ibvs.constants import *

# ============================================================
# Camera → Body transform (VERIFY THIS FOR YOUR RIG)
# ============================================================
R_CB = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0]
], dtype=float)

P_CB = np.array([P_CB_X, P_CB_Y, P_CB_Z])


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

        # ---------------- Publisher ----------------
        self.rc_pub = self.create_publisher(
            OverrideRCIn,
            "/mavros/rc/override",
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

        # ---------------- Gains (TUNE IN WATER) ----------------
        self.K_SURGE = 300     # PWM per m/s
        self.K_SWAY  = 300
        self.K_HEAVE = 350
        self.K_YAW   = 220

        self.HEAVE_BIAS = 40   # buoyancy trim (PWM)

        # ---------------- Desired image features ----------------
        self.desired_pts = self.compute_desired_corners(
            Z_DES, FX, FY, CX, CY, TAG_SIZE
        )

        self.create_timer(0.1, self.tag_watchdog)

        self.get_logger().info("IBVS RC-Override Controller READY (EKF BYPASSED)")

    # =========================================================
    # Desired image points (static setpoint at Z_DES)
    # =========================================================
    def compute_desired_corners(self, Z_des, fx, fy, cx, cy, tag_size):
        s = tag_size / 2.0
        corners_3d = np.array([
            [-s, -s, Z_des],
            [ s, -s, Z_des],
            [ s,  s, Z_des],
            [-s,  s, Z_des],
        ])

        pts = np.zeros((4, 2), dtype=float)
        for i, (X, Y, Z) in enumerate(corners_3d):
            pts[i, 0] = fx * X / Z + cx
            pts[i, 1] = fy * Y / Z + cy
        return pts

    # =========================================================
    def cb_depth(self, msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001

    # =========================================================
    # Correct IBVS interaction matrix (point feature + depth)
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
        pwm = 1500 + gain * v + bias
        return int(np.clip(pwm, 1100, 1900))

    # =========================================================
    def cb_corners(self, msg):
        if self.depth_img is None or len(msg.polygon.points) != 4:
            return

        self.last_tag_time = self.get_clock().now()
        self.tag_lost = False

        rows = []
        errs = []

        h, w = self.depth_img.shape

        for i, p in enumerate(msg.polygon.points):
            u, v = p.x, p.y
            ui, vi = int(u), int(v)

            if not (0 <= ui < w and 0 <= vi < h):
                return

            patch = self.depth_img[
                max(0, vi-PATCH):min(h, vi+PATCH+1),
                max(0, ui-PATCH):min(w, ui+PATCH+1)
            ]

            valid = patch[patch > 0]
            if valid.size == 0:
                return

            Z = float(np.median(valid))
            if Z < 0.3 or Z > 6.0:
                return

            rows.append(self.interaction_matrix(u, v, Z))

            x, y = (u - CX)/FX, (v - CY)/FY
            xd, yd = (self.desired_pts[i] - [CX, CY]) / [FX, FY]
            errs.extend([x - xd, y - yd, Z - Z_DES])

        L = np.vstack(rows)
        e = np.array(errs).reshape(-1, 1)

        # ---------------- IBVS law ----------------
        mu = 0.01
        Vc = -LAMBDA_P * np.linalg.inv(
            L.T @ L + mu * np.eye(6)
        ) @ L.T @ e

        v_c = Vc[0:3]
        w_c = Vc[3:6]

        # ---------------- Camera → Body ----------------
        Wb = R_CB @ w_c
        Vb = (R_CB @ v_c) + np.cross(Wb.flatten(), P_CB).reshape(3, 1)

        # ---------------- RC Override ----------------
        rc = OverrideRCIn()
        rc.channels = [65535] * 18

        rc.channels[4] = self.vel_to_pwm(Vb[0,0], self.K_SURGE)               # Surge
        rc.channels[5] = self.vel_to_pwm(Vb[1,0], self.K_SWAY)                # Sway
        rc.channels[2] = self.vel_to_pwm(Vb[2,0], self.K_HEAVE, self.HEAVE_BIAS)
        rc.channels[3] = self.vel_to_pwm(Wb[2,0], self.K_YAW)                 # Yaw

        self.rc_pub.publish(rc)

        # ---------------- Debug error ----------------
        err_msg = Float32MultiArray()
        err_msg.data = np.array(errs, dtype=np.float32).tolist()
        self.err_pub.publish(err_msg)

    # =========================================================
    def tag_watchdog(self):
        if self.last_tag_time is None:
            return

        now = self.get_clock().now()
        dt = (now - self.last_tag_time).nanoseconds * 1e-9

        if dt > self.TAG_TIMEOUT and not self.tag_lost:
            self.tag_lost = True
            self.get_logger().warn("AprilTag LOST → stopping thrusters")

            rc = OverrideRCIn()
            rc.channels = [1500] * 18
            self.rc_pub.publish(rc)


# =============================================================
def main():
    rclpy.init()
    rclpy.spin(IBVSRCController())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
