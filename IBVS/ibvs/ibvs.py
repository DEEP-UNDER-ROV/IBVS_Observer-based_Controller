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

from ibvs.constants import *

class IBVSRCController(Node):
    def __init__(self):
        super().__init__("ibvs_rc_controller")

        self.bridge = CvBridge()

        # ---------------- Subscribers ----------------
        self.sub_corners = self.create_subscription(PolygonStamped, "/apriltag/corners", self.cb_corners,qos_profile_sensor_data)
        self.sub_detection = self.create_subscription(AprilTagDetectionArray,"/detection1",self.cb_detection,10)

        # ---------------- Publishers ----------------
        self.rc_override_pub = self.create_publisher(OverrideRCIn, "/mavros/rc/override", 10)
        self.pwm_pub = self.create_publisher(Int16MultiArray, "/ibvs/pwm_debug", 10)
        self.vel_cam_pub = self.create_publisher(TwistStamped, "/ibvs/vel_cam", 10)
        self.vel_body_pub = self.create_publisher(TwistStamped, "/ibvs/vel_body", 10)
        self.err_pub = self.create_publisher(Float32MultiArray, "/ibvs/error", 10)

        # ---------------- State ----------------
        self.current_pwm = [1500] * 18
        self.depth_img = None
        self.detected_uv = None
        self.last_tag_time = None
        
        # PI or SMC controller state
        self.e_integral = None
        self.ev_integral = None
        self.last_time = None
        self.sta_state = None


        ##Tuneable Variables
        self.HEAVE_BIAS = 0        
        self.lambda_smc = 2.0          # Sliding surface parameter = to lambda gain in PI
        self.phi = 0.02                # Boundary layer thickness

        # Sliding gains 
        # Higher gain produces faster convergence but more chattering, lower gain produce smooth motion, same like \mu in DLS
        self.Ks = np.diag([0.8, 0.8, 0.8, 0.0, 0.3, 0.0])

        # K1 act like P gain, K2 act like disturbance estimator
        self.K1 = np.diag([0.9, 0.9, 0.9, 0.0, 0.35, 0.0])
        self.K2 = np.diag([0.25, 0.25, 0.25, 0.0, 0.08, 0.0])

                            # Sway - Heave - Surge - Pitch - Yaw - Roll
        self.declare_parameter("Kp", [0.1,0.5,0.3,0.7,0.3,0.7]) 
        self.declare_parameter("Ki", [0.0,0.03,0.0,0.0,0.03,0.0])
        
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

        self.desired_pts = self.compute_desired_corners_pixel(Z_DES)


    # # =========================================================
    # def compute_desired_corners(self, Z_DES):
    #     half = TAG_SIZE / 2.0
    #     corners = np.array([
    #         [-half, -half],
    #         [ half, -half],
    #         [ half,  half],
    #         [-half,  half],
    #     ])
    #     desired = corners / Z_DES
    #     return desired
    
    # =========================================================
    # # Compute desired using u,v in /detection1/
    # # def compute_desired_corners(self, FX, FY, CX, CY):
    # def compute_desired_corners(self, Z_DES):

    #     half = TAG_SIZE/2

    #     corners = np.array([
    #         [-half, half, Z_DES],
    #         [ half, half, Z_DES],
    #         [ half,-half, Z_DES],
    #         [-half,-half, Z_DES],
    #     ])

    #     desired = []

    #     for X,Y,Z in corners:

    #         u = FX * X/Z + CX
    #         v = FY * Y/Z + CY

    #         desired.append([u,v])

    #     return np.array(desired)

    # =========================================================
    def compute_desired_corners_pixel(self, Z_DES):
        half = TAG_SIZE / 2.0

        # Tag corners in tag frame (meters)
        corners = np.array([
            [-half,  half],   # top-left
            [ half,  half],   # top-right
            [ half, -half],   # bottom-right
            [-half, -half],   # bottom-left
        ])

        # Normalize
        x = corners[:, 0] / Z_DES
        y = corners[:, 1] / Z_DES

        # Convert to pixels
        u = FX * x + CX
        v = FY * y + CY

        desired_pixels = np.column_stack((u, v))

        return desired_pixels

    # # =========================================================
    # def interaction_matrix(self, x, y, Z):
    #     return np.array([
    #         [-1/Z,  0,    x/Z,      x*y,     -(1 + x*x),  y],
    #         [0,    -1/Z,  y/Z,    1 + y*y,      -x*y,    -x],
    #         [0,     0, -1/Z_DES, -y*Z/Z_DES,  x*Z/Z_DES,  0]
    #     ])
    
    # # =========================================================
    # # Compute L matrix using u,v in /detection1/ 
    # def interaction_matrix_pixel(self, u, v, Z):

    #     du = u - CX
    #     dv = v - CY

    #     return np.array([

    #         [-FX/Z, 0,     du/Z,     du*dv/FY,         -(FX+du**2/FX),  FX*dv/FY],

    #         [0,     -FY/Z, dv/Z,     FY+dv**2/FY,      -du*dv/FX,       -FY*du/FX],

    #         [0,     0,     -1/Z_DES, -Z*dv/(FY*Z_DES), Z*du/(FX*Z_DES), 0]
    #     ])

    # =========================================================
    def interaction_matrix_pixel(self, u, v, Z):
        x = (u - CX) / FX
        y = (v - CY) / FY

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
    def cb_detection(self, msg):

        self.get_logger().info(
            f"detection_uv:\n{self.detected_uv}",
            throttle_duration_sec=1.0
        )

        if len(msg.detections) == 0:
            self.detected_uv = None
            return

        det = msg.detections[0]

        self.detected_uv = np.array(
            [[c.x, c.y] for c in det.corners],
            dtype=np.float64
        )

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
        pixel_norm = []
        measured_pts = []

        pts = np.array([[p.x, p.y, p.z] for p in msg.polygon.points])

        self.get_logger().info(
            f"polygon:\n{pts}",
            throttle_duration_sec=1.0
        )

        # Function to use the pixel error from left_cam instead of norm error
        for i in range(4):
            u,v = self.detected_uv[i]
            Z = pts[i,2]

            # v = -v

            ud,vd = self.desired_pts[i]
            rows.append(self.interaction_matrix_pixel(u,v,Z))

            measured_pts.append([u, v])

            z_norm = (Z - Z_DES) / Z_DES

            errs.extend([u-ud, v-vd, z_norm])
            pixel_norm.extend([u - ud, v - vd])
                    
        L = np.vstack(rows)
        e = np.array(errs).reshape(-1, 1)

        # ---------- Time step ----------
        now = self.get_clock().now()
        
        if self.last_time is None:
            dt = 0.04
        else:
            dt = (now - self.last_time).nanoseconds * 1e-9
        
        self.last_time = now

        # ------- Solve pseudo-inverse term ----------
        A = L.T @ L + mu**2 * np.eye(6)

        L_pinv = np.linalg.pinv(L)
        e_v = L_pinv @ e

        # ------- Initialize PI or SMC states ---------
        if self.e_integral is None:
            self.e_integral = np.zeros_like(e)

        if self.ev_integral is None:
            self.ev_integral = np.zeros_like(e_v)

        # ---------- Integral of image error ---------- anti windup cek lgi
        self.e_integral += e * dt
        self.e_integral = np.clip(self.e_integral, -0.3, 0.3)

        self.ev_integral += e_v * dt
        self.ev_integral = np.clip(self.ev_integral, -0.3, 0.3)

        # # ----- First Order SMC Control Law -----
        # s = e_v + self.lambda_smc * self.ev_integral
        # sat = np.clip(s / self.phi, -1.0, 1.0)
        # Vc = - (self.lambda_smc * e_v + self.Ks @ sat)


        # # ----- Second Order SMC Control Law -----
        # s = e_v + self.lambda_smc * self.ev_integral
        # if self.sta_state is None:
        #     self.sta_state = np.zeros_like(e_v)

        # sign_s = np.sign(s)
        # self.sta_state += sign_s * dt

        # sqrt_s = np.sqrt(np.abs(s) + 1e-8)

        # Vc = -(self.lambda_smc * e_v + self.K1 @ (sqrt_s * sign_s) + self.K2 @ self.sta_state)


        # # ------------ PI Control Law ------------
        # b_p = L.T @ e
        # b_i = L.T @ self.e_integral

        # v_p = np.linalg.solve(A, b_p).flatten()
        # v_i = np.linalg.solve(A, b_i).flatten()

        # Vc = -(self.Kp_mat @ v_p + self.Ki_mat @ v_i)

        # dead_band = np.abs(pixel_norm)
        # if np.max(dead_band) < 5.0:
        #     Vc[:] = 0.0

        Vc[0:3] = np.clip(Vc[0:3], -MAX_LIN_VEL, MAX_LIN_VEL)
        Vc[3:6] = np.clip(Vc[3:6], -MAX_ANG_VEL, MAX_ANG_VEL)

        v_c = Vc[0:3]
        w_c = Vc[3:6]

        v_c = v_c.reshape(3,)
        w_c = w_c.reshape(3,)
        
        Wb = (R_BC @ w_c).reshape(3,)
        Vb = (R_BC @ v_c).reshape(3,) + np.cross(Wb, P_BC.reshape(3,))
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

        # self.get_logger().info(
        #     f"Cam_PosX = {Vc[0]:.3f} |"
        #     f"Cam_PosY = {Vc[1]:.3f} |"
        #     f"Cam_PosZ = {Vc[2]:.3f} |"
        #     f"Cam_RotX = {Vc[3]:.3f} |"
        #     f"Cam_RotY = {Vc[4]:.3f} |"
        #     f"Cam_RotZ = {Vc[5]:.3f} |",
        #     throttle_duration_sec=0.5
        # )
        
        self.get_logger().info(
            f"Surge = {Vb[0]:.3f} |"
            f"Sway = {Vb[1]:.3f} |"
            f"Heave = {Vb[2]:.3f} |"
            f"Roll = {Wb[0]:.3f} |"
            f"Pitch = {Wb[1]:.3f} |"
            f"Yaw = {Wb[2]:.3f} ",
            throttle_duration_sec=0.5
        )

        # self.get_logger().info(f"Corners: {pts}")

        # -------- Compute PWM --------
        pwm = [1500] * 18
        
        pwm[4] = self.vel_to_pwm(Vb[0], K_SURGE)
        pwm[5] = self.vel_to_pwm(Vb[1], K_SWAY)
        pwm[2] = self.vel_to_pwm(Vb[2], K_HEAVE, self.HEAVE_BIAS)
        pwm[1] = self.vel_to_pwm(Wb[0], K_ROLL)
        pwm[0] = self.vel_to_pwm(Wb[1], K_PITCH)
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
            f"Roll(RC2) = {pwm[1]} |"
            f"Pitch(RC1) = {pwm[0]} |"
            f"Yaw(RC4) = {pwm[3]}",
            throttle_duration_sec=0.5
        )

        # self.get_logger().info(
        #     f"x={x:.3f} "
        #     f"xd={xd:.3f} "
        #     f"err={x-xd:.3f} "
        #     f"Z={Z:.3f}",
        #     throttle_duration_sec=0.2
        # )
        
        # self.get_logger().info(
        #     f"||e||={np.linalg.norm(e):.4f}, "
        #     f"||LTe||={np.linalg.norm(L.T @ e):.4f}",
        #     throttle_duration_sec=0.2
        # )

        # measured_pts = np.array(measured_pts)
        # self.get_logger().info(
        #     f"\nMeasured:\n{np.round(measured_pts, 3)}"
        #     f"\nDesired:\n{np.round(self.desired_pts, 3)}",
        #     throttle_duration_sec=0.5
        # )

        for i, (m, d) in enumerate(zip(measured_pts, self.desired_pts)):
            self.get_logger().info(
                f"P{i+1}: "
                f"M=({m[0]:.3f}, {m[1]:.3f}) "
                f"D=({d[0]:.3f}, {d[1]:.3f})",
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
                self.get_logger().warn("AprilTag LOST")
                self.e_integral = None
                self.last_time = None
            
        #     # Only reset to neutral IF the tag is actually lost
            self.current_pwm = [1500] * 18
            self.current_pwm[2] = 1500 + 100      # RC3 (Heave), +200 bias

            self.get_logger().info(
            f"Tag lost: Heave PWM = {self.current_pwm[2]}",
            throttle_duration_sec=0.5
        )

# =============================================================
def main():
    rclpy.init()
    node = IBVSRCController()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
