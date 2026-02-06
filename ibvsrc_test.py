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

R_CB = np.array([[0,0,1],
                 [1,0,0],
                 [0,1,0]], dtype=float)

P_CB = np.array([P_CB_X, P_CB_Y, P_CB_Z])


class IBVSRCController(Node):
    def __init__(self):
        super().__init__("ibvs_rc_controller")

        self.bridge = CvBridge()

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

        self.rc_pub = self.create_publisher(
            OverrideRCIn,
            "/mavros/rc/override",
            10
        )

        self.depth_img = None

        self.K_SURGE = 600
        self.K_SWAY  = 600
        self.K_YAW   = 300

        self.HEAVE_BIAS = 0

        self.desired_pts = self.compute_desired_corners(
            Z_DES, FX, FY, CX, CY, TAG_SIZE
        )

        self.get_logger().info("IBVS RC Controller ACTIVE")

    # ------------------------------------------------
    def compute_desired_corners(self, Z, fx, fy, cx, cy, tag_size):
        s = tag_size / 2.0
        pts = []
        for X, Y in [(-s,-s),(s,-s),(s,s),(-s,s)]:
            pts.append([fx*X/Z + cx, fy*Y/Z + cy])
        return np.array(pts)

    def cb_depth(self, msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001

    # ------------------------------------------------
    def interaction_matrix(self, u, v, Z):
        x = (u - CX)/FX
        y = (v - CY)/FY
        return np.array([
            [-1/Z, 0, x/Z,  x*y, -(1+x*x), y],
            [0, -1/Z, y/Z, 1+y*y, -x*y, -x]
        ])

    # ------------------------------------------------
    def vel_to_pwm(self, v, gain):
        if abs(v) < 0.01:
            v = 0.01 * np.sign(v) if v != 0 else 0.01
        return int(np.clip(1500 + gain * v, 1100, 1900))

    # ------------------------------------------------
    def cb_corners(self, msg):
        if self.depth_img is None or len(msg.polygon.points) != 4:
            return

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
            Z = float(np.median(patch[patch > 0]))
            if Z < 0.3:
                return

            rows.append(self.interaction_matrix(u, v, Z))

            x, y = (u - CX)/FX, (v - CY)/FY
            xd, yd = (self.desired_pts[i]-[CX,CY])/[FX,FY]
            errs.extend([x - xd, y - yd])

        L = np.vstack(rows)
        e = np.array(errs).reshape(-1,1)

        Vc = -LAMBDA_P * np.linalg.pinv(L) @ e

        v_c = Vc[0:3]
        w_c = Vc[3:6]

        Wb = R_CB @ w_c
        Vb = R_CB @ v_c

        self.get_logger().info(
            f"Vb = [{Vb[0,0]:+.3f}, {Vb[1,0]:+.3f}]  "
            f"Wb_yaw = {Wb[2,0]:+.3f}",
            throttle_duration_sec=0.3
        )

        rc = OverrideRCIn()
        rc.channels = [65535]*18
        rc.channels[4] = self.vel_to_pwm(Vb[0,0], self.K_SURGE)
        rc.channels[5] = self.vel_to_pwm(Vb[1,0], self.K_SWAY)
        rc.channels[3] = self.vel_to_pwm(Wb[2,0], self.K_YAW)

        self.rc_pub.publish(rc)


def main():
    rclpy.init()
    rclpy.spin(IBVSRCController())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
