#IBVS Z_err Normalized

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import numpy as np

from apriltag_msgs.msg import AprilTagDetectionArray
from geometry_msgs.msg import PolygonStamped, TwistStamped
from std_msgs.msg import Float32MultiArray, Int16MultiArray
from mavros_msgs.msg import OverrideRCIn
from rcl_interfaces.msg import SetParametersResult
from cv_bridge import CvBridge

from .parameter import *

class IBVSRCController(Node):
    def __init__(self):
        super().__init__("ibvs_rc_controller")

        self.bridge = CvBridge()

        # ---------------- Subscribers ----------------
        self.sub_corners = self.create_subscription(PolygonStamped, "/apriltag/corners", self.cb_corners,qos_profile_sensor_data)
        self.sub_detection = self.create_subscription(AprilTagDetectionArray, "/detection1", self.cb_detection, 10)

        # ---------------- Publishers ----------------
        self.rc_override_pub = self.create_publisher(OverrideRCIn, "/mavros/rc/override", 10)
        self.pwm_pub = self.create_publisher(Int16MultiArray, "/ibvs/pwm_debug", 10)
        self.vel_cam_pub = self.create_publisher(TwistStamped, "/ibvs/vel_cam", 10)
        self.vel_body_pub = self.create_publisher(TwistStamped, "/ibvs/vel_body", 10)
        self.err_pub = self.create_publisher(Float32MultiArray, "/ibvs/error", 10)

        # ---------------- State ----------------
        self.current_pwm = [1500] * 18
        self.detected_uv = None
        self.depth_img = None
        self.last_tag_time = None
        
        # PID state
        self.v_prev = None
        self.e_integral = None
        self.v_dot = None
        self.last_time = None

        self.declare_parameter("Kp", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        self.declare_parameter("Ki", [0.0, 0.0, 0.0, 0.0, 0.2, 0.0])
        
        self.Kp = np.array(self.get_parameter("Kp").value)
        self.Ki = np.array(self.get_parameter("Ki").value)
        
        self.Kp_mat = np.diag(self.Kp)
        self.Ki_mat = np.diag(self.Ki)

        self.T_bc = self.transform_matrix(P_BC_0, R_CLB)
        
        self.add_on_set_parameters_callback(self.param_callback)
        
        self.tag_lost = True
        self.TAG_TIMEOUT = 0.5  # seconds

        # RC command buffer (IMPORTANT)
        self.rc_cmd = [1500] * 18

        # ---------------- Timers ----------------
        self.create_timer(0.1, self.tag_watchdog)
        self.create_timer(1.0/25.0, self.publish_rc)
        self.get_logger().info("IBVS Control Active")

        self.desired_pts, R = self.desired_corners(Z_DES=Z_DES, pitch_deg=PITCH_DES_DEG, yaw_deg=YAW_DES_DEG, roll_deg=ROLL_DES_DEG)

    # =========================================================
    def desired_corners(self, Z_DES, pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0):
        half = TAG_SIZE / 2.0

        corners = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0]
        ], dtype=float)

        rx, ry, rz = np.deg2rad(pitch_deg), np.deg2rad(yaw_deg), np.deg2rad(roll_deg)

        Rx = np.array([[1,0,0],
                    [0,np.cos(rx),-np.sin(rx)],
                    [0,np.sin(rx), np.cos(rx)]])

        Ry = np.array([[ np.cos(ry),0,np.sin(ry)],
                    [0,1,0],
                    [-np.sin(ry),0,np.cos(ry)]])

        Rz = np.array([[np.cos(rz),-np.sin(rz),0],
                    [np.sin(rz), np.cos(rz),0],
                    [0,0,1]])

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
    def interaction_matrix(self, x, y, Z):
        return np.array([
            [-1/Z,  0,  x/Z,   x*y,  -(1 + x*x),  y],
            [0,   -1/Z, y/Z, 1 + y*y,   -x*y,    -x],
            [0,     0, -1/Z,   -y,        x,      0]
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
    def vel_to_pwm(self, v, bias=0):
        return int(np.clip(1500 + 400 * v + bias, 1100, 1900))

    # =========================================================
    def cb_detection(self, msg):
        if len(msg.detections) == 0:
            self.detected_uv = None
            return

        det = msg.detections[0]
        self.detected_uv = np.array([[c.x, c.y] for c in det.corners],dtype=np.float64)

    # =========================================================   
    def pixel_to_norm(self, u, v):
        x = (u - CX)/FX
        y = (v - CY)/FY
        return x, y
    
    # =========================================================
    def cb_corners(self, msg):
        if self.detected_uv is None:
            return
                
        if len(msg.polygon.points) != 4:
            return

        self.last_tag_time = self.get_clock().now()
        self.tag_lost = False

        rows = []
        errs = []
        pixel_err = []

        pts = np.array([[p.x, p.y, p.z] for p in msg.polygon.points])
        for i in range(4):
            u, v = self.detected_uv[i]
            Z = pts[i, 2]
            if Z <= 0:
                return
            
            # Desired normalized
            x, y = self.pixel_to_norm(u, v)

            ud, vd = self.desired_pts[i]
            xd, yd = self.pixel_to_norm(ud, vd)
            
            rows.append(self.interaction_matrix(x, y, Z))
            errs.extend([x - xd, y - yd, Z - Z_DES])
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
        L_pinv = np.linalg.solve(L.T @ L + mu**2 * np.eye(6), L.T)
        
        # ----------- Initialize PID states -----------
        if self.e_integral is None:
            self.e_integral = np.zeros(e.shape[0])

        self.e_integral += e.flatten() * dt
        self.e_integral = np.clip(self.e_integral, -0.2, 0.2)

        v_P = L_pinv @ e.reshape(-1)
        v_I = L_pinv @ self.e_integral
        
        # ----------------- PID Control -----------------
        Vc = - (self.Kp_mat @ v_P) - (self.Ki_mat @ v_I)

        pixel_err_magnitude = np.abs(pixel_err)
        if np.max(pixel_err_magnitude) < 0.005:
            Vc[:] = 0.0

        Vb = self.T_bc @ Vc
        Vb[2] = -Vb[2]

        # ------------- ROS2 MSG VCam -------------
        vel_body_msg = TwistStamped()
        vel_body_msg.header.stamp = msg.header.stamp
        vel_body_msg.header.frame_id = "body"
        
        vel_body_msg.twist.linear.x = float(Vb[0])
        vel_body_msg.twist.linear.y = float(Vb[1])
        vel_body_msg.twist.linear.z = float(Vb[2])
        vel_body_msg.twist.angular.x = float(Vb[3])
        vel_body_msg.twist.angular.y = float(Vb[4])
        vel_body_msg.twist.angular.z = float(Vb[5])
        self.vel_body_pub.publish(vel_body_msg)
        
        self.get_logger().info(
                f"Surge = {Vb[0]:.2f} |"
                f"Sway = {Vb[1]:.2f} |"
                f"Heave = {Vb[2]:.2f} |"
                f"Roll = {Vb[3]:.2f} |"
                f"Pitch = {Vb[4]:.2f} |"
                f"Yaw = {Vb[5]:.2f} ",
                throttle_duration_sec=1.0)

        # -------- Compute PWM --------
        pwm = [1500] * 18
        
        pwm[4] = self.vel_to_pwm(Vb[0])
        pwm[5] = self.vel_to_pwm(Vb[1])
        pwm[2] = self.vel_to_pwm(Vb[2])
        pwm[1] = self.vel_to_pwm(Vb[3])
        pwm[0] = self.vel_to_pwm(Vb[4])
        pwm[3] = self.vel_to_pwm(Vb[5])
        self.current_pwm = pwm 
                
        # Publish debug PWM
        msg_pwm = Int16MultiArray()
        msg_pwm.data = pwm
        self.pwm_pub.publish(msg_pwm)

        self.get_logger().info(
                f"Surge = {pwm[4]} |"
                f"Sway = {pwm[5]} |"
                f"Heave = {pwm[2]} |"
                f"Roll = {pwm[1]} |"
                f"Pitch = {pwm[0]} |"
                f"Yaw = {pwm[3]}",
                throttle_duration_sec=1.0)

        err_msg = Float32MultiArray()
        err_msg.data = np.array(e, dtype=np.float32).flatten().tolist()
        self.err_pub.publish(err_msg)

    # =========================================================
    def skew(self, p):
        return np.array([
            [0,-p[2],p[1]],
            [p[2],0,-p[0]],
            [-p[1],p[0],0]
        ])

    # =========================================================
    def transform_matrix(self, P_BC, R_CB):
        S = self.skew(P_BC)
        T_bc = np.block([
            [R_CB,           -R_CB @ S],
            [np.zeros((3,3)),     R_CB]])

        return T_bc

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
