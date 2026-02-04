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

    def compute_desired_corners(self, Z_des, fx, fy, cx, cy, tag_size):
        s = TAG_SIZE / 2.0
    
        # Tag corners in camera frame (meters)
        corners_3d = np.array([
            [-s, -s, Z_des],
            [ s, -s, Z_des],
            [ s,  s, Z_des],
            [-s,  s, Z_des],
        ])
    
        desired = np.zeros((4, 2), dtype=np.float32)
    
        for i, (X, Y, Z) in enumerate(corners_3d):
            u = FX * (X / Z) + CX
            v = FY * (Y / Z) + CY
            desired[i] = [u, v]
    
        return desired

    # -------------------------------------------------

    def cb_pnp(self, msg):
        self.p_pnp[:] = [msg.x, msg.y, msg.z]

    def cb_ekf(self, msg):
        self.current_pose = msg
        q = msg.pose.orientation
        self.yaw = math.atan2(2*q.w*q.z, 1 - 2*q.z*q.z)

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

        for i, p in enumerate(msg.polygon.points):
            u, v, Z = p.x, p.y, p.z
            if Z <= 0.2:
                return

            rows.append(self.interaction_matrix(u, v, Z))

            x, y = (u - CX)/FX, (v - CY)/FY
            xd, yd = (self.desired_pts[i] - [CX, CY]) / [FX, FY]
            errs.extend([x - xd, y - yd, Z - Z_DES])

        L = np.vstack(rows)
        e = np.array(errs).reshape(-1, 1)

        mu = 0.01
        Vc = -LAMBDA_P * np.linalg.inv(L.T @ L + mu * np.eye(6)) @ L.T @ e

        v_c = Vc[0:3]
        w_c = Vc[3:6]

        Wb = R_CB @ w_c
        Vb = (R_CB @ v_c) + np.cross(Wb.flatten(), P_CB).reshape(3, 1)

        # --- IBVS INTEGRATION ---
        self.p_corr += Vb.flatten() * dt
        self.p_corr = np.clip(self.p_corr, -MAX_OFFSET, MAX_OFFSET)

        p_cmd = self.p_pnp + self.p_corr

        sp = PoseStamped()
        sp.header.stamp = now.to_msg()
        sp.header.frame_id = "map"

        sp.pose.position.x = p_cmd[0]
        sp.pose.position.y = p_cmd[1]
        sp.pose.position.z = self.current_pose.pose.position.z
        # sp.pose.position.z = p_cmd[2]

        sp.pose.orientation = self.current_pose.pose.orientation

        self.sp_pub.publish(sp)

        err_msg = Float32MultiArray()
        err_msg.data = errs
        self.err_pub.publish(err_msg)


def main():
    rclpy.init()
    rclpy.spin(IBVSPnPPositionController())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
