#IBVS Z_err Normalized

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import numpy as np

from geometry_msgs.msg import PolygonStamped, TwistStamped
from std_msgs.msg import Float32MultiArray, Int16MultiArray
from mavros_msgs.msg import OverrideRCIn
from rcl_interfaces.msg import SetParametersResult
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
        self.vel_cam_pub = self.create_publisher(TwistStamped, "/ibvs/vel_cam", 10)
        self.vel_body_pub = self.create_publisher(TwistStamped, "/ibvs/vel_body", 10)
        self.err_pub = self.create_publisher(Float32MultiArray, "/ibvs/error", 10)

        # ---------------- State ----------------
        self.current_pwm = [1500] * 18
        self.depth_img = None
        self.last_tag_time = None
        
        # PID state
        self.v_prev = None
        self.v_integral = None
        self.v_dot = None
        self.last_time = None

        self.declare_parameter("Kp", [0.7,0.7,0.5,0.0,1.2,0.0])
        self.declare_parameter("Ki", [0.0,0.0,0.0,0.0,0.2,0.0])
        
        self.Kp = np.array(self.get_parameter("Kp").value)
        self.Ki = np.array(self.get_parameter("Ki").value)
        
        self.Kp_mat = np.diag(self.Kp)
        self.Ki_mat = np.diag(self.Ki)
        
        self.add_on_set_parameters_callback(self.param_callback)
        
        self.tag_lost = True
        self.TAG_TIMEOUT = 0.5  # seconds

        # RC command buffer (IMPORTANT)
        self.rc_cmd = [1500] * 18

        # ---------------- Timers ----------------
        self.create_timer(0.1, self.tag_watchdog)
        self.create_timer(1.0/25.0, self.publish_rc)
        self.get_logger().info("IBVS Control Active")

        self.desired_pts = self.compute_desired_corners(Z_DES)

    # =========================================================
    def compute_desired_corners(self, Z_des):
        half = TAG_SIZE / 2.0
        corners = np.array([
            [-half, -half],
            [ half, -half],
            [ half,  half],
            [-half,  half],
        ])
        desired = corners / Z_des
        return desired

    # =========================================================
    def reorder_corners_ccw(self, pts):
        pts = np.array(pts)
    
        cx = np.mean(pts[:,0])
        cy = np.mean(pts[:,1])
    
        angles = np.arctan2(pts[:,1] - cy, pts[:,0] - cx)
        order = np.argsort(angles)
    
        pts = pts[order]
    
        # rotate so first point is bottom-left
        idx = np.argmin(pts[:,0] + pts[:,1])
        pts = np.roll(pts, -idx, axis=0)
        return pts

    # =========================================================
    def interaction_matrix(self, x, y, Z):
        return np.array([
            [-1/Z,  0,    x/Z,      x*y,     -(1 + x*x),  y],
            [0,    -1/Z,  y/Z,    1 + y*y,      -x*y,    -x],
            [0,     0, -1/Z_DES, -y*Z/Z_DES,  x*Z/Z_DES,  0]
        ])

    # =========================================================
    def param_callback(self, params):
        for param in params:
            if param.name == "Kp":
                self.Kp = np.array(param.value)
                self.Kp_mat = np.diag(self.Kp)
            
            elif param.name == "Ki":
                self.Ki = np.array(param.value)
                self.Ki_mat = np.diag(self.Ki)
                            
        return SetParametersResult(successful=True)
        
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

        pts = np.array([[p.x, p.y, p.z] for p in msg.polygon.points])
        pts = self.reorder_corners_ccw(pts)
        for i in range(4):
            x, y, Z = pts[i]
            if Z <= 0:
                return
            
            # Desired normalized
            xd, yd = self.desired_pts[i]
            
            # Build interaction matrix using normalized coords
            rows.append(self.interaction_matrix(x, y, Z))
            
            # Error vector
            z_norm = (Z - Z_DES) / Z_DES
            errs.extend([x - xd, y - yd, z_norm])
            
            # Stop condition error
            pixel_err.extend([x - xd, y - yd])
            
        L = np.vstack(rows)
        e = np.array(errs).reshape(-1, 1)

        # ---------- Time step ----------
        now = self.get_clock().now()
        
        if self.last_time is None:
            dt = 0.04
        else:
            dt = (now - self.last_time).nanoseconds * 1e-9
        
        self.last_time = now

        # ------------ Solve IBVS velocity ------------
        A = L.T @ L + mu**2 * np.eye(6)
        b = L.T @ e
        v_raw = -np.linalg.solve(A, b).flatten()
        
        # ----------- Initialize PID states -----------
        if self.v_prev is None:
            self.v_prev = np.zeros(6)
            self.v_integral = np.zeros(6)
            self.v_dot = np.zeros(6)
        
        # ------------- PID terms Integral -------------
        self.v_integral += v_raw * dt
        self.v_integral = np.clip(self.v_integral, -0.2, 0.2)

        # ------------ PID terms Derivative ------------
        alpha = 0.7
        v_dot_raw = (v_raw - self.v_prev) / dt
        self.v_dot = alpha * self.v_dot + (1 - alpha) * v_dot_raw
        
        # ----------------- PID Control -----------------
        Vc = (self.Kp_mat @ v_raw) + (self.Ki_mat @ self.v_integral)
        self.v_prev = v_raw

        pixel_err_magnitude = np.abs(pixel_err)
        if np.max(pixel_err_magnitude) < 0.005:
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

        # ------------- ROS2 MSG VCam -------------
        vel_cam_msg = TwistStamped()
        vel_cam_msg.header.stamp = msg.header.stamp
        vel_cam_msg.header.frame_id = "camera"
        
        vel_cam_msg.twist.linear.x = float(Vc[0])
        vel_cam_msg.twist.linear.y = float(Vc[1])
        vel_cam_msg.twist.linear.z = float(Vc[2])
        vel_cam_msg.twist.angular.x = float(Vc[3])
        vel_cam_msg.twist.angular.y = float(Vc[4])
        vel_cam_msg.twist.angular.z = float(Vc[5])
        self.vel_cam_pub.publish(vel_cam_msg)

        # ------------- ROS2 MSG VBody -------------
        vel_body_msg = TwistStamped()
        vel_body_msg.header.stamp = msg.header.stamp
        vel_body_msg.header.frame_id = "body"
        
        vel_body_msg.twist.linear.x = float(Vb[0])
        vel_body_msg.twist.linear.y = float(Vb[1])
        vel_body_msg.twist.linear.z = float(Vb[2])
        vel_body_msg.twist.angular.x = float(Wb[0])
        vel_body_msg.twist.angular.y = float(Wb[1])
        vel_body_msg.twist.angular.z = float(Wb[2])
        self.vel_body_pub.publish(vel_body_msg)

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
        
        pwm[4] = self.vel_to_pwm(Vb[0], K_SURGE)
        pwm[5] = self.vel_to_pwm(Vb[1], K_SWAY)
        pwm[2] = self.vel_to_pwm(Vb[2], K_HEAVE, HEAVE_BIAS)
        pwm[3] = self.vel_to_pwm(Wb[2], K_YAW)

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

    # =========================================================
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
                self.e_prev = None
                self.e_integral = None
                self.last_time = None
            
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
