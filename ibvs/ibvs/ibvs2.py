#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PolygonStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from mavros_msgs.msg import PositionTarget

from ibvs.constants import (
    FX, FY, CX, CY,
    DESIRED_SIZE, LAMBDA_P, PATCH,
    Z_DES, P_CB_X, P_CB_Y, P_CB_Z,
    MAX_LIN_VEL, MAX_ANG_VEL
)

# Camera → Body rotation
R_CB = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0]
], dtype=float)

p_CB = np.array([P_CB_X, P_CB_Y, P_CB_Z])


class IBVSControllerNode(Node):
    def __init__(self):
        super().__init__("ibvs_controller")

        self.bridge = CvBridge()

        self.sub_corners = self.create_subscription(
            PolygonStamped, "/apriltag/corners", self.cb_corners, 10
        )
        self.sub_depth = self.create_subscription(
            Image, "/camera/depth/image_raw", self.cb_depth, 10
        )

        self.raw_pub = self.create_publisher(
            PositionTarget, "/mavros/setpoint_raw/local", 10
        )
        self.err_pub = self.create_publisher(
            Float32MultiArray, "/ibvs/error", 10
        )

        self.depth_img = None

        s = DESIRED_SIZE // 2
        self.desired_pts = np.array([
            [CX - s, CY - s],
            [CX + s, CY - s],
            [CX + s, CY + s],
            [CX - s, CY + s]
        ], dtype=np.float32)

        self.get_logger().info("IBVS Controller (RAW velocity mode) started")

    def cb_depth(self, msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001

    def interaction_matrix(self, u, v, Z):
        x = (u - CX) / FX
        y = (v - CY) / FY
        return np.array([
            [-1/Z, 0, x/Z, x*y, -(1+x*x), y],
            [0, -1/Z, y/Z, 1+y*y, -x*y, -x],
            [0, 0, -1, -y*Z, x*Z, 0]
        ])

    def cb_corners(self, msg):
        if self.depth_img is None:
            return

        pts = np.array([[p.x, p.y] for p in msg.polygon.points])
        rows, errs = [], []

        h, w = self.depth_img.shape

        for i in range(4):
            u, v = pts[i]
            ui, vi = int(round(u)), int(round(v))

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
            rows.append(self.interaction_matrix(u, v, Z))

            curr_x = (u - CX) / FX
            curr_y = (v - CY) / FY
            des_x = (self.desired_pts[i, 0] - CX) / FX
            des_y = (self.desired_pts[i, 1] - CY) / FY

            errs.extend([curr_x - des_x, curr_y - des_y, Z - Z_DES])

        Ls = np.vstack(rows)
        e = np.array(errs).reshape(-1, 1)

        Vc = -LAMBDA_P * (np.linalg.pinv(Ls) @ e)

        v_c = Vc[0:3].reshape(3, 1)
        w_c = Vc[3:6].reshape(3, 1)

        w_b = R_CB @ w_c
        v_b = (R_CB @ v_c) + np.cross(w_b.flatten(), p_CB).reshape(3, 1)

        vx = float(np.clip(v_b[0], -MAX_LIN_VEL, MAX_LIN_VEL))
        vy = float(np.clip(v_b[1], -MAX_LIN_VEL, MAX_LIN_VEL))

        # ⚠️ NED: positive Z is DOWN
        vz = float(np.clip(v_b[2], -MAX_LIN_VEL, MAX_LIN_VEL))

        wz = float(np.clip(w_b[2], -MAX_ANG_VEL, MAX_ANG_VEL))

        # ---------------- RAW SETPOINT ----------------
        msg = PositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.coordinate_frame = PositionTarget.FRAME_BODY_NED

        # Velocity + yaw_rate only
        msg.type_mask = 1479

        msg.velocity.x = vx
        msg.velocity.y = vy
        msg.velocity.z = vz   # +down

        msg.yaw_rate = wz

        self.raw_pub.publish(msg)

        self.get_logger().info(
            f"vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}, yaw_rate={wz:.2f}"
        )

        err_msg = Float32MultiArray()
        err_msg.data = np.array(errs, dtype=np.float32).flatten().tolist()
        self.err_pub.publish(err_msg)


def main():
    rclpy.init()
    node = IBVSControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
