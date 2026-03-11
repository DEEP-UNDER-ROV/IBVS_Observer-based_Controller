#IBVS Z_err Normalized

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import numpy as np

from geometry_msgs.msg import PolygonStamped, TwistStamped
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
        self.sub_corners = self.create_subscription(PolygonStamped, "/apriltag/corners", self.cb_corners,qos_profile_sensor_data)

        # ---------------- Publishers ----------------
        self.rc_override_pub = self.create_publisher(OverrideRCIn, "/mavros/rc/override", 10)
        self.pwm_pub = self.create_publisher(Int16MultiArray, "/ibvs/pwm_debug", 10)
        self.vel_pub = self.create_publisher(TwistStamped, "/ibvs/vel", 10)
        self.err_pub = self.create_publisher(Float32MultiArray, "/ibvs/error", 10)

        # ---------------- State ----------------
        self.current_pwm = [1500] * 18
        self.depth_img = None
        self.last_tag_time = None
        self.tag_lost = True
        self.TAG_TIMEOUT = 0.5  # seconds

        # RC command buffer (IMPORTANT)
        self.rc_cmd = [1500] * 18

        # ---------------- Gains (TUNE IN WATER) ----------------
        self.K_SURGE = 500
        self.K_SWAY  = 1200
        self.K_HEAVE = 750
        self.K_YAW   = 200
        self.HEAVE_BIAS = -133

        # ---------------- Desired image features ----------------
        self.desired_pts = self.compute_desired_corners(
            Z_DES, FX, FY, CX, CY, TAG_SIZE
        )

        # ---------------- Timers ----------------
        self.create_timer(0.1, self.tag_watchdog)
        self.create_timer(1.0/25.0, self.publish_rc)
        self.get_logger().info("IBVS Control Active")

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
    def interaction_matrix(self, u, v, Z):
        x = (u - CX) / FX
        y = (v - CY) / FY
        return np.array([
            [-1/Z,  0,    x/Z,      x*y,     -(1 + x*x),  y],
            [0,    -1/Z,  y/Z,    1 + y*y,      -x*y,    -x],
            [0,     0, -1/Z_DES, -y*Z/Z_DES,  x*Z/Z_DES,  0]
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

        for i, p in enumerate(msg.polygon.points):
            u, v, Z = p.x, p.y, p.z
            if Z <= 0:
                return
                
            ud, vd = self.desired_pts[i]
            pixel_err.extend([u - ud, v - vd])
            rows.append(self.interaction_matrix(u, v, Z))

            x, y = (u - CX)/FX, (v - CY)/FY
            
            ud = self.desired_pts[i,0]
            vd = self.desired_pts[i,1]
            
            xd = (ud - CX) / FX
            yd = (vd - CY) / FY
            
            z_norm = (Z - Z_DES) / Z_DES
            errs.extend([x - xd, y - yd, z_norm])
            print("Actual:", p.x, p.y)
            print("Desired:", self.desired_pts[i])
            # errs.extend([x - xd, y - yd, 0.25*(Z - Z_DES)])

        L = np.vstack(rows)
        e = np.array(errs).reshape(-1, 1)

        A = L.T @ L + mu**2 * np.eye(6)
        b = L.T @ e
        Vc = - np.diag(LAMBDA_P) @ np.linalg.solve(A,b)

        pixel_err_magnitude = np.abs(pixel_err)
        if np.max(pixel_err_magnitude) < 1.5:
            Vc[:] = 0.0

        Vc[0:3] = np.clip(Vc[0:3], -MAX_LIN_VEL, MAX_LIN_VEL)
        Vc[3:6] = np.clip(Vc[3:6], -MAX_ANG_VEL, MAX_ANG_VEL)

        v_c = Vc[0:3]
        w_c = Vc[3:6]

        v_c = v_c.reshape(3,)
        w_c = w_c.reshape(3,)
        
        Wb = (R_CB @ w_c).reshape(3,)
        Vb = (R_CB @ v_c).reshape(3,) + np.cross(Wb, P_CB.reshape(3,))
        Vb[2] = -Vb[2]

        vel_msg = TwistStamped()
        vel_msg.header.stamp = msg.header.stamp
        vel_msg.header.frame_id = "body"
        vel_msg.twist.linear.x = float(Vb[0])
        vel_msg.twist.linear.y = float(Vb[1])
        vel_msg.twist.linear.z = float(Vb[2])
        vel_msg.twist.angular.x = float(Wb[0])
        vel_msg.twist.angular.y = float(Wb[1])
        vel_msg.twist.angular.z = float(Wb[2])
        self.vel_pub.publish(vel_msg)

        self.get_logger().info(
            f"Cam_PosX = {Vc[0]} |"
            f"Cam_PosY = {Vc[1]} |"
            f"Cam_PosZ = {Vc[2]} |"
            f"Cam_RotX = {Vc[3]} |"
            f"Cam_RotY = {Vc[4]} |"
            f"Cam_RotZ = {Vc[5]} |",
            throttle_duration_sec=0.5
        )
        
        self.get_logger().info(
            f"Surge = {Vb[0]} |"
            f"Sway = {Vb[1]} |"
            f"Heave = {Vb[2]} |"
            f"Yaw Body = {Wb[2]}",
            throttle_duration_sec=0.5
        )

        # -------- Compute PWM --------
        pwm = [1500] * 18
        
        pwm[4] = self.vel_to_pwm(Vb[0], self.K_SURGE)
        pwm[5] = self.vel_to_pwm(Vb[1], self.K_SWAY)
        pwm[2] = self.vel_to_pwm(Vb[2], self.K_HEAVE, self.HEAVE_BIAS)
        pwm[3] = self.vel_to_pwm(Wb[2], self.K_YAW)

        self.current_pwm = pwm 
        
        # Publish debug PWM
        msg_pwm = Int16MultiArray()
        msg_pwm.data = pwm
        self.pwm_pub.publish(msg_pwm)

        self.get_logger().info(
            f"Surge(RC5) = {pwm[4]} |"
            f"Sway(RC6) = {pwm[5]} |"
            f"Heave(RC3) = {pwm[2]} |"
            f"Yaw(RC4) = {pwm[3]}",
            throttle_duration_sec=0.5
        )

        err_msg = Float32MultiArray()
        err_msg.data = np.array(e, dtype=np.float32).flatten().tolist()
        self.err_pub.publish(err_msg)

    def publish_rc(self):
        rc_msg = OverrideRCIn()
        rc_msg.channels = [int(c) for c in self.current_pwm]
        self.rc_override_pub.publish(rc_msg)
        
    # =========================================================
    def tag_watchdog(self):

        if self.last_tag_time is None:
            return

        dt = (self.get_clock().now() - self.last_tag_time).nanoseconds * 1e-9

        if dt > self.TAG_TIMEOUT:
            if not self.tag_lost:
                self.tag_lost = True
                self.get_logger().warn("AprilTag LOST → Neutral RC")
            
            # Only reset to neutral IF the tag is actually lost
            self.current_pwm = [1500] * 18

# =============================================================
def main():
    rclpy.init()
    node = IBVSRCController()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
