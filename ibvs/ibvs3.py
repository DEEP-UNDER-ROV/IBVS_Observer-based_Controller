#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math

from geometry_msgs.msg import (
    PolygonStamped,
    PoseStamped,
    Point
)
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from ibvs.constants import *

qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT
)


class IBVSPnPPositionController(Node):
    def __init__(self):
        super().__init__("ibvs_pnp_position_controller")

        # Subscribers
        self.sub_corners = self.create_subscription( PolygonStamped, "/apriltag/corners", self.cb_corners, 10)
        self.sub_pnp = self.create_subscription( Point, "/pnp/relative_position", self.cb_pnp, 10)
        # self.sub_ekf = self.create_subscription(PoseStamped, "/mavros/local_position/pose", self.cb_ekf, 10)
        self.sub_pose = self.create_subscription(PoseStamped, "/mavros/local_position/pose", self.cb_ekf, qos)

        # Publisher
        self.sp_pub = self.create_publisher( PoseStamped, "/mavros/setpoint_position/local", 10)
        self.err_pub = self.create_publisher( Float32MultiArray, "/ibvs/error", 10)

        self.desired_pts = self.compute_desired_corners( Z_DES, FX, FY, CX, CY, TAG_SIZE)

        self.p_pnp = np.zeros(3)
        self.p_corr = np.zeros(3)
        self.last_time = self.get_clock().now()
        self.current_pose = None
        self.yaw = 0.0

        self.get_logger().info("IBVS + PnP POSITION controller running")

    # -------------------------------------------------

    @staticmethod
    def compute_desired_corners(Z_DES, FX, FY, CX, CY, TAG_SIZE):
        s = TAG_SIZE / 2.0
    
        # Tag corners in camera frame (meters)
        corners_3d = np.array([
            [-s, -s, Z_DES],
            [ s, -s, Z_DES],
            [ s,  s, Z_DES],
            [-s,  s, Z_DES],
        ])
    
        desired = np.zeros((4, 2), dtype=np.float32)
    
        for i, (X, Y, Z) in enumerate(corners_3d):
            u = FX * (X / Z) + CX
            v = FY * (Y / Z) + CY
            desired[i] = [u, v]
    
        return desired

    # -------------------------------------------------

    def cb_pnp(self, msg):
        alpha = 0.2
        new = np.array([msg.x, msg.y, msg.z])
        self.p_pnp = alpha * new + (1 - alpha) * self.p_pnp

    def cb_ekf(self, msg):
        self.current_pose = msg
        q = msg.pose.orientation
        siny_cosp = 2 * (q.w*q.z + q.x*q.y)
        cosy_cosp = 1 - 2 * (q.y*q.y + q.z*q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    # -------------------------------------------------

    def interaction_matrix(self, u, v, Z):
        x = (u - CX) / FX
        y = (v - CY) / FY
        return np.array([
            [-1/Z,  0,    x/Z,   x*y,     -(1 + x*x),  y],
            [0,    -1/Z,  y/Z,   1 + y*y, -x*y,       -x],
            [0,     0,   -1,    -y*Z,      x*Z,        0]
        ])

    # -------------------------------------------------

    def cb_corners(self, msg):
        if len(msg.polygon.points) != 4 or self.current_pose is None:
            return

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        if dt <= 0:
            return

        rows, errs = [], []
        pixel_err = []

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

        aligned = np.mean(np.abs(pixel_err)) < 10.0

        if aligned:
            self.p_corr *= 0.0
            p_cmd = self.p_pnp
            return

        v_c = Vc[0:3]
        w_c = Vc[3:6]

        v_c = v_c.reshape(3,)
        w_c = w_c.reshape(3,)
        
        Wb = (R_CB @ w_c).reshape(3,)
        Vb = (R_CB @ v_c).reshape(3,) + np.cross(Wb, P_CB.reshape(3,))

        # --- IBVS INTEGRATION ---
        # ---- DT LIMITING ----
        dt = np.clip(dt, 0.0, 0.05)   # max 50 ms step
        
        # ---- VELOCITY SATURATION ----
        Vb_limited = np.clip(Vb, -MAX_LIN_VEL, MAX_LIN_VEL)
        
        # ---- ANTI-WINDUP BACK CALCULATION ----
        Kaw = 0.5   # anti-windup gain (tune 0.1–1.0)
        
        # ---- LEAKAGE (prevents drift over time) ----
        leak = 0.02  # small decay factor
        
        self.p_corr += (Vb_limited + Kaw*(Vb_limited - Vb)) * dt
        self.p_corr -= leak * self.p_corr * dt
        
        # ---- HARD LIMIT ----
        self.p_corr = np.clip(self.p_corr, -MAX_OFFSET, MAX_OFFSET)

        R_yaw = np.array([
            [math.cos(self.yaw), -math.sin(self.yaw), 0],
            [math.sin(self.yaw),  math.cos(self.yaw), 0],
            [0, 0, 1]
        ])
        
        p_corr_map = R_yaw @ self.p_corr
        p_cmd = self.p_pnp + p_corr_map

        sp = PoseStamped()
        sp.header.stamp = now.to_msg()
        sp.header.frame_id = "map"

        sp.pose.position.x = p_cmd[0]
        sp.pose.position.y = p_cmd[1]
        # sp.pose.position.z = self.current_pose.pose.position.z
        sp.pose.position.z = p_cmd[2]

        sp.pose.orientation = self.current_pose.pose.orientation

        self.get_logger().info(
            "\n".join([f"P{i+1}: {ex:+.4f} {ey:+.4f} {ez:+.4f}"
            for i, (ex, ey, ez) in enumerate(err_array)]),
            throttle_duration_sec=0.5
        )

        self.get_logger().info(
            f"IBVS OUT | "
            f"Vb [m/s]: x={Vb[0]:+.3f}, y={Vb[1]:+.3f}, z={Vb[2]:+.3f} | "
            f"p_corr [m]: x={self.p_corr[0]:+.3f}, y={self.p_corr[1]:+.3f}, z={self.p_corr[2]:+.3f} | "
            f"p_pnp [m]: x={self.p_pnp[0]:+.3f}, y={self.p_pnp[1]:+.3f}, z={self.p_pnp[2]:+.3f} | "
            f"p_cmd [m]: x={p_cmd[0]:+.3f}, y={p_cmd[1]:+.3f}, z={p_cmd[2]:+.3f}",
            throttle_duration_sec=0.5
        )

        self.sp_pub.publish(sp)

        err_msg = Float32MultiArray()
        err_msg.data = errs
        self.err_pub.publish(err_msg)


def main():
    rclpy.init()
    node = IBVSVelocityController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
