#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import PolygonStamped, Twist, Point
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ibvs.constants import *


class IBVSControllerNode(Node):
    def __init__(self):
        super().__init__("IBVSControllerNode")
        self.bridge = CvBridge()

        # Subscribers
        self.sub_corners = self.create_subscription(PolygonStamped, "/apriltag/corners", self.cb_corners, 10)
        self.sub_depth = self.create_subscription(Image, "/camera/depth/image_raw", self.cb_depth, 10)
        self.sub_pnp = self.create_subscription(Point, "/pnp/relative_position", self.cb_pnp, 10)

        # Publishers
        self.vel_pub = self.create_publisher(Twist, "/ibvs/vel", 10)
        self.pos_pub = self.create_publisher(Point, "/ibvs/pos", 10)
        self.err_pub = self.create_publisher(Float32MultiArray, "/ibvs/error", 10)

        self.depth_img = None
        self.last_time = self.get_clock().now()

        # Desired image features
        self.desired_pts = self.desired_corners(Z_DES, FX, FY, CX, CY, TAG_SIZE)

        # --- Dynamic extension state ---
        self.p_hat = np.zeros(3)       # virtual IBVS position
        self.p_pnp = np.zeros(3)       # anchor (from PnP)

        # --- Tag visibility handling ---
        self.last_tag_time = None
        self.tag_lost = True
        self.TAG_TIMEOUT = 0.5     # seconds
        self.DECAY_RATE = 0.9
        self.tag_lost = False

        self.create_timer(0.1, self.timer_check_tag)
        self.get_logger().info("CAUTION !! IBVS Control ON")

    # ---------------------------------------------------------

    def desired_corners(self, Z_DES , fx, fy, cx, cy, tag_size):
        half = tag_size / 2.0
        corners = np.array([
            [-half, -half, Z_DES],
            [ half, -half, Z_DES],
            [ half,  half, Z_DES],
            [-half,  half, Z_DES],
        ])

        pts = np.zeros((4, 2), dtype=float)
    
        for i, (X, Y, Z) in enumerate(corners):
            u = fx * (X / Z) + cx
            v = fy * (Y / Z) + cy
            pts[i] = [u, v]
        return pts

    # ---------------------------------------------------------

    def cb_depth(self, msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001

    def cb_pnp(self, msg):
        # Relative pose anchor (tag frame)
        self.p_pnp[:] = np.array([msg.x, msg.y, msg.z])

    # ---------------------------------------------------------

    def interaction_matrix(self, u, v, Z):
        x = (u - CX) / FX
        y = (v - CY) / FY
        return np.array([
            [-1/Z, 0.0,  x/Z,  x*y, -(1+x*x),  y],
            [0.0, -1/Z,  y/Z,  1+y*y, -x*y,   -x]
        ])

    # ---------------------------------------------------------

    def cb_corners(self, msg):
        if self.depth_img is None:
            return

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        if dt <= 0:
            return

        self.last_tag_time = rclpy.time.Time.from_msg(msg.header.stamp)
        self.tag_lost = False

        if self.tag_lost:
            return

        rows = []
        errs = []

        pts = np.array([[p.x, p.y] for p in msg.polygon.points])
        h, w = self.depth_img.shape

        for i, (u, v) in enumerate(pts):
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
            if Z < 0.2:
                return

            # normalized coordinates
            rows.append(self.interaction_matrix(u, v, Z))
            
            x, y = (u - CX)/ FX, (v - CY)/ FY
            xd, yd = (self.desired_pts[i]-[CX,CY])/[FX,FY]
            errs.extend([x - xd, y - yd])

        # --- IBVS control law ---
        L = np.vstack(rows)                  # 8x6
        e = np.array(errs).reshape(-1, 1)    # 8x1

        mu = 1
        Vc = -LAMBDA_P * np.linalg.inv(
            L.T @ L + mu * np.eye(6)) @ L.T @ e

        v_c = Vc[0:3]
        w_c = Vc[3:6]

        # Camera → body
        Wb = R_CB @ w_c
        Vb = (R_CB @ v_c) + np.cross(Wb.flatten(), P_CB).reshape(3, 1)

        # --- Dynamic extension (anchored integrator) ---
        v_ibvs = Vb.flatten()
        p_err = self.p_hat - self.p_pnp
        self.p_hat += (v_ibvs - K_ANCHOR * p_err) * dt
        self.p_hat = np.clip(self.p_hat, -MAX_OFFSET, MAX_OFFSET)

        # Publish position command
        self.pos_pub.publish(Point(
            x=float(self.p_hat[0]),
            y=float(self.p_hat[1]),
            z=float(self.p_hat[2])
        ))

        # (Optional) publish velocity for logging
        vel = Twist()
        vel.linear.x = float(np.clip(Vb[0].item(), -MAX_LIN_VEL, MAX_LIN_VEL))
        vel.linear.y = float(np.clip(Vb[1].item(), -MAX_LIN_VEL, MAX_LIN_VEL))
        vel.linear.z = float(np.clip(Vb[2].item(), -MAX_LIN_VEL, MAX_LIN_VEL))
        vel.angular.x = float(np.clip(Wb[0].item(), -MAX_ANG_VEL, MAX_ANG_VEL))
        vel.angular.y = 0.0 
        vel.angular.z = float(np.clip(Wb[2].item(), -MAX_ANG_VEL, MAX_ANG_VEL))
        self.vel_pub.publish(vel)

        # Error logging
        err_msg = Float32MultiArray()
        err_msg.data = np.array(errs, dtype=np.float32).tolist()
        self.err_pub.publish(err_msg)

    def timer_check_tag(self):
        if self.last_tag_time is None:
            return

        now = self.get_clock().now()
        lost_dt = (now - self.last_tag_time).nanoseconds * 1e-9

        if lost_dt > self.TAG_TIMEOUT and not self.tag_lost:
            self.tag_lost = True
            self.get_logger().warn("AprilTag LOST → zero IBVS control")

            # Decay internal state
            self.p_hat *= self.DECAY_RATE

            # Zero velocity
            self.vel_pub.publish(Twist())

            # Hold / decay position
            self.pos_pub.publish(Point(
                x=float(self.p_hat[0]),
                y=float(self.p_hat[1]),
                z=float(self.p_hat[2])
            ))
        else:
            self.tag_lost = False

def main():
    rclpy.init()
    node = IBVSControllerNode()
    node.create_timer(0.05, node.timer_check_tag)  # 20 Hz safety check
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
