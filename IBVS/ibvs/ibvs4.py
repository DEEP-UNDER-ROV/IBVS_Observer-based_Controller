#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy)
import numpy as np

from apriltag_msgs.msg import AprilTagDetectionArray
from geometry_msgs.msg import PolygonStamped, TwistStamped, WrenchStamped
from std_msgs.msg import Float32MultiArray, Int16MultiArray
from mavros_msgs.msg import OverrideRCIn
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Imu
from cv_bridge import CvBridge

from parameter import *

class IBVSRCController(Node):
    def __init__(self):
        super().__init__("ibvs_rc_controller")

        self.bridge = CvBridge()
        imu_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=20)
        
        # ---------------- Subscribers ----------------
        self.sub_corners = self.create_subscription(PolygonStamped, "/apriltag/corners", self.cb_corners, qos_profile_sensor_data)
        self.sub_detection = self.create_subscription(AprilTagDetectionArray, "/detection1", self.cb_detection, 10)
        # self.camera_gyro_sub = self.create_subscription(Imu, '/camera/camera/gyro/sample', self.cb_camera_gyro, 100)
        # self.camera_accel_sub = self.create_subscription(Imu, '/camera/camera/accel/sample', self.cb_camera_accel, 100)
        self.fcu_imu_sub = self.create_subscription(Imu, '/mavros/imu/data', self.cb_fcu_imu, imu_qos)

        # ---------------- Publishers ----------------
        self.rc_override_pub = self.create_publisher(OverrideRCIn, "/mavros/rc/override", 10)
        self.pwm_pub = self.create_publisher(Int16MultiArray, "/ibvs/pwm_debug", 10)

        self.vel_cam_pub = self.create_publisher(TwistStamped, "/ibvs/vel_cam", 10)
        self.vel_body_pub = self.create_publisher(TwistStamped, "/ibvs/vel_body", 10)

        self.nu_B_hat_pub = self.create_publisher(TwistStamped, "/ibvs/nu_B_hat", 10)
        self.torque_pub = self.create_publisher(WrenchStamped, "/ibvs/torque", 10)
        self.tau_P_pub = self.create_publisher(WrenchStamped, "/ibvs/torque/p", 10)
        self.tau_D_pub = self.create_publisher(WrenchStamped, "/ibvs/torque/d", 10)
        self.tau_L_pub = self.create_publisher(WrenchStamped, "/ibvs/torque/l", 10)
        self.tau_gamma_pub = self.create_publisher(WrenchStamped, "/ibvs/torque/gamma", 10)

        self.ukf_data_pub = self.create_publisher(Float32MultiArray, "/ibvs/ukf/data", 10)
        self.err_px_pub = self.create_publisher(Float32MultiArray, "/ibvs/error/px", 10)
        self.err_no_pub = self.create_publisher(Float32MultiArray, "/ibvs/error/no", 10)

        # ---------------- State ----------------
        self.use_3d_matrix_feature = True
        self.ukf_initialized = False
        self.use_camera_ukf = True
        self.camera_measurement_valid = False

        # self.compute_control = self.compute_control_ibvs1
        self.freeze_L_alpha = True

        self.L_frozen = None
        self.alpha_frozen = None
        self.alpha_pinv_frozen = None

        self.freeze_initialized = False

        self.Minv = np.linalg.inv(M)

        self.current_pwm = [1500] * 18
        self.depth_img = None
        self.detected_uv = None
        self.last_tag_time = None
        self.last_imu_time = None
        self.last_camera_time = None
        self.last_control_time = None
        self.L_prev = None
        self.L_hat = None
        self.L_dot = None
        self.last_delta = None
        
        # PI controller state
        self.last_time = None
        self.e_integral = None

        # Camera IMU
        self.acc_camera = None
        self.acc_camera_B = None
        self.gyro_camera = None
        self.acc_camera_stamp = None
        self.gyro_camera_stamp = None

        # FCU IMU
        self.acc_fcu = None
        self.acc_fcu_B = None
        self.gyro_fcu = None
        self.acc_fcu_stamp = None
        self.gyro_fcu_stamp = None

        ##Tuneable Variables
        self.HEAVE_BIAS = 0 

                     # Sway - Heave - Surge - Pitch - Yaw - Roll
        self.Kp = np.diag([0.7,0.2,0.6,0.3,0.2,0.3]) 
        self.Kd = np.diag([0.02,0.0,0.0,0.01,0.01,0.01])
        
        self.tag_lost = True
        self.TAG_TIMEOUT = 1  # seconds

        self.T_bc = self.camera_body_adjoint()

        # ---------------- Timers ----------------
        self.create_timer(0.1, self.tag_watchdog)
        self.create_timer(1.0/25.0, self.publish_rc)
        self.control_timer = self.create_timer(1.0 / 30.0, self.cb_control)
        self.get_logger().info(f"IBVS Control {'3D Matrix' if self.use_3d_matrix_feature else '2D Matrix'}")

        self.desired_pts, R = self.compute_desired_corners_pixel(Z_DES=Z_DES, pitch_deg=PITCH_DES_DEG, yaw_deg=YAW_DES_DEG, roll_deg=ROLL_DES_DEG)

        self.desired_normal = R @ np.array([0.0, 0.0, 1.0])

        p0 = self.desired_pts[3]   # bottom-left
        p1 = self.desired_pts[2]   # bottom-right
        self.desired_roll = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
        self.desired_pitch = np.arctan2(-self.desired_normal[1], self.desired_normal[2])
        self.desired_yaw = np.arctan2(self.desired_normal[0], self.desired_normal[2])

        self.N = 4
        self.n_ft = 3 * self.N if self.use_3d_matrix_feature else 2 * self.N
        self.nx = self.n_ft + 18

        self.idx_sC = slice(0, self.n_ft)
        self.idx_vB = slice(self.n_ft, self.n_ft + 3)
        self.idx_wB = slice(self.n_ft + 3, self.n_ft + 6)
        self.idx_bg = slice(self.n_ft + 6, self.n_ft + 9)
        self.idx_aB = slice(self.n_ft + 9, self.n_ft + 12)
        self.idx_ba = slice(self.n_ft + 12, self.n_ft + 15)
        self.idx_bo = slice(self.n_ft + 15, self.n_ft + 18)

        self.idx_nuB = np.concatenate([
            np.arange(self.idx_vB.start, self.idx_vB.stop),
            np.arange(self.idx_wB.start, self.idx_wB.stop)
        ])

        # ---------- UKF Parameters ----------
        self.alpha = 0.5
        self.beta = 2.0
        self.kappa = 0.0

        self.q_u = 1e-6
        self.q_v = 1e-6
        self.q_Z = 1e-5

        self.q_vB = 1e-3
        self.q_wB = 1e-3

        self.q_aB = 1e-2

        self.sigma_bg_rw = 7.692845772051322e-7
        self.sigma_ba_rw = 0.003597694747410243
        self.sigma_bo_rw = 1e-4

        self.R_camera = np.eye(self.n_ft) * 1e-4

        self.lambda_ = self.alpha**2 * (self.nx + self.kappa) - self.nx
        self.gamma = np.sqrt(self.nx + self.lambda_)
        self.n_sigma = 2 * self.nx + 1

        # ---------- Weights ----------
        self.Wm = np.full(2*self.nx+1,
                          1.0/(2*(self.nx+self.lambda_)))

        self.Wc = np.full(2*self.nx+1,
                          1.0/(2*(self.nx+self.lambda_)))

        self.Wm[0] = self.lambda_/(self.nx+self.lambda_)
        self.Wc[0] = self.Wm[0] + (1-self.alpha**2+self.beta)

        self.camera_imu_timeshift = 0.00702

        self.ukf_x = np.zeros((self.nx,1))
        self.ukf_P = np.eye(self.nx)*1e-3
        self.ukf_R = np.eye(self.n_ft)*1e-4

        self.tau_ukf = np.zeros((6,1))

        self.s_hat = np.zeros(self.n_ft)
        self.nu_B_hat = np.zeros((6,1))
        self.nu_C_hat = np.zeros((6,1))
        self.b_a_hat = np.zeros((3,1))
        self.b_g_hat = np.zeros((3,1))

    @property
    def feature_dim(self):
        return 3 * self.N if self.use_3d_matrix_feature else 2 * self.N

    def stamp_to_sec(self, stamp):
        return (float(stamp.sec) + float(stamp.nanosec) * 1e-9)

    # =========================================================
    def pack_state(self, s_C, v_B, w_B, b_g, a_B, b_a, b_o): 
        state = np.zeros(self.nx, dtype=np.float64)

        state[self.idx_sC] = np.asarray(s_C, dtype=np.float64).reshape(-1)
        state[self.idx_vB] = np.asarray(v_B, dtype=np.float64).reshape(-1)
        state[self.idx_wB] = np.asarray(w_B, dtype=np.float64).reshape(-1)
        state[self.idx_bg] = np.asarray(b_g, dtype=np.float64).reshape(-1)
        state[self.idx_aB] = np.asarray(a_B, dtype=np.float64).reshape(-1)
        state[self.idx_ba] = np.asarray(b_a, dtype=np.float64).reshape(-1)
        state[self.idx_bo] = np.asarray(b_o, dtype=np.float64).reshape(-1)

        return state
    
    # =========================================================
    def unpack_state(self, state):
        state = np.asarray(state, dtype=np.float64).reshape(-1)

        s_C = state[self.idx_sC].copy()
        v_B = state[self.idx_vB].copy()
        w_B = state[self.idx_wB].copy()
        b_g = state[self.idx_bg].copy()
        a_B = state[self.idx_aB].copy()
        b_a = state[self.idx_ba].copy()
        b_o = state[self.idx_bo].copy()

        return s_C, v_B, w_B, b_g, a_B, b_a, b_o

    # =========================================================
    def validate_camera_measurement(self, z_camera):
        z_camera = np.asarray(z_camera, dtype=np.float64).reshape(self.n_ft)

        if not np.all(np.isfinite(z_camera)):
            return False

        for i in range(self.N):
            idx = 3 * i
            x = z_camera[idx]
            y = z_camera[idx + 1]
            Z = z_camera[idx + 2]

            if Z <= 1e-4:
                return False

            if Z > 20.0:
                return False

            if abs(x) > 2.0:
                return False

            if abs(y) > 2.0:
                return False

        return True

    # =========================================================
    def cb_fcu_imu(self, msg):
        q = msg.orientation
        t = self.stamp_to_sec(msg.header.stamp)

        if self.last_imu_time is None:
            self.last_imu_time = t
            R_NB = self.quaternion_to_rotation(msg.orientation)

            return

        dt = t - self.last_imu_time
        self.last_imu_time = t

        if dt <= 0.0 or dt > 0.1:
            return

        R_NB = self.quaternion_to_rotation(msg.orientation)

        accel_flu = np.array([msg.linear_acceleration.x, 
                              msg.linear_acceleration.y, 
                              msg.linear_acceleration.z])

        gyro_flu = np.array([msg.angular_velocity.x, 
                             msg.angular_velocity.y, 
                             msg.angular_velocity.z])

        accel_B, gyro_B = self.imu_flu_to_body(accel_flu, gyro_flu)

        z_imu = np.concatenate([accel_B, gyro_B])

        if not self.ukf_initialized:
            return

        x_pred, P_pred, sigma_pred = self.ukf_predict(self.ukf_x, self.ukf_P, dt, None)
        self.ukf_x, self.ukf_P, imu_innovation, S_imu, K_imu, z_imu_mean = self.ukf_update_imu(x_pred, P_pred, sigma_pred, z_imu, R_NB)

        # self.log_covariance_blocks(self.ukf_P)

        # P = 0.5 * (self.ukf_P + self.ukf_P.T)
        # eigvals = np.linalg.eigvalsh(P)

        # self.get_logger().info(
        #     f"UKF P after IMU: "
        #     f"min_eig={eigvals.min():.6e}, "
        #     f"max_eig={eigvals.max():.6e}, "
        #     f"trace={np.trace(P):.6e}, "
        #     f"asym={np.linalg.norm(self.ukf_P - self.ukf_P.T):.6e}"
        # )

        # self.get_logger().info(
        #     f"UKF P finite = {np.all(np.isfinite(self.ukf_P))}"
        # )
        
        self.ukf_logging(source="imu", innovation=imu_innovation, K=K_imu, z=z_imu)

        self.sC_hat = self.ukf_x[self.idx_sC].copy()
        self.vB_hat = self.ukf_x[self.idx_vB].copy()
        self.wB_hat = self.ukf_x[self.idx_wB].copy()
        self.bg_hat = self.ukf_x[self.idx_bg].copy()
        self.aB_hat = self.ukf_x[self.idx_aB].copy()
        self.ba_hat = self.ukf_x[self.idx_ba].copy()
        self.bo_hat = self.ukf_x[self.idx_bo].copy()

        self.nu_B_hat = np.concatenate([self.vB_hat, self.wB_hat]).reshape(6, 1)
        self.nu_C_hat = self.b_to_c_velocity(self.nu_B_hat)

        self.last_imu_innovation = imu_innovation.copy()

        self.publish_twist(self.nu_B_hat_pub, "nu_B_hat", msg.header.stamp, self.vB_hat, self.wB_hat)
        
        self.get_logger().info("UKF velocity = [{}]".format(
                ", ".join(f"{v:.3f}" for v in self.nu_B_hat.flatten())),
                throttle_duration_sec=1.0)
        
    # =========================================================
    def cb_corners(self, msg):
        if self.detected_uv is None:
            return
    
        if len(msg.polygon.points) != self.N:
            return
        
        camera_time = self.stamp_to_sec(msg.header.stamp)
        self.last_camera_time = camera_time

        self.last_tag_time = self.get_clock().now()
        self.tag_lost = False

        result = self.compute_image_error(msg)
        if result is None:
            return

        distance, e_pixel, e_norm, measurement, e_pixel_img, deltas = result
        z_camera = measurement.flatten()

        ## Camera noise
        # z_camera = self.add_camera_measurement_noise(z_camera_raw)
        # self.last_camera_measurement_raw = z_camera_raw.copy()
        # self.last_camera_measurement = z_camera.copy()

        self.publish_error(e_pixel, e_norm)

        if not self.validate_camera_measurement(z_camera):
            self.get_logger().warn("Invalid stereo feature measurement.")
            return

        if not self.ukf_initialized:
            self.initialize_ukf_from_camera(z_camera)

            self.sC_hat = self.ukf_x[self.idx_sC].copy()
            self.vB_hat = self.ukf_x[self.idx_vB].copy()
            self.wB_hat = self.ukf_x[self.idx_wB].copy()
            self.get_logger().info("UKF initialized from stereo camera.")

            return

        if self.use_camera_ukf:
            sigma_camera = self.generate_sigma_points(self.ukf_x, self.ukf_P)
        
            self.ukf_x, self.ukf_P, cam_innovation, S_cam, K_cam, z_cam_mean = self.ukf_update_camera(self.ukf_x, self.ukf_P, sigma_camera, z_camera)

            self.ukf_logging(source="camera", innovation=cam_innovation, K=K_cam, z=z_camera)

            self.last_camera_innovation = cam_innovation.copy()

        self.latest_distance = distance
        self.latest_e_norm = e_norm.copy()
        self.latest_e_pixel = e_pixel_img.copy()

        self.camera_measurement_valid = True

    # =========================================================
    def cb_control(self):
        if not self.ukf_initialized:
            return

        if not self.camera_measurement_valid:
            return

        now = self.get_clock().now()
        if self.last_control_time is None:
            self.last_control_time = now
            return

        dt = (now - self.last_control_time).nanoseconds * 1e-9

        self.last_control_time = now

        if dt <= 0.0 or dt > 0.2:
            return

        distance = self.latest_distance
        e_norm = self.latest_e_norm.copy()
        e_pixel = self.latest_e_pixel.copy()
    
        tau = self.compute_control_tau_DLS(distance, e_norm, e_pixel, dt)

        self.tau_ukf = tau.reshape(6,1)
        self.publish_torque(self.torque_pub, "body", self.get_clock().now().to_msg(), tau)
        self.publish_torque(self.tau_P_pub, "body", self.get_clock().now().to_msg(), self.tau_P)
        self.publish_torque(self.tau_D_pub, "body", self.get_clock().now().to_msg(), self.tau_D)
        self.publish_torque(self.tau_L_pub, "body", self.get_clock().now().to_msg(), self.tau_L)
        self.publish_torque(self.tau_gamma_pub, "body", self.get_clock().now().to_msg(), self.tau_gamma)

        pwm = self.compute_force_pwm(tau)
        
        self.publish_rc()
        self.log_debug(tau, pwm)


        # self.sC_hat = self.ukf_x[self.idx_sC].copy()
        # self.vB_hat = self.ukf_x[self.idx_vB].copy()
        # self.wB_hat = self.ukf_x[self.idx_wB].copy()
        # self.nu_B_hat = np.concatenate([self.vB_hat, self.wB_hat]).reshape(6, 1)
        # self.b_g_hat = self.ukf_x[self.idx_bg].copy()
        # self.a_B_hat = self.ukf_x[self.idx_aB].copy()
        # self.b_a_hat = self.ukf_x[self.idx_ba].copy()
        # self.b_o_hat = self.ukf_x[self.idx_bo].copy()

        # self.nu_C_hat = self.b_to_c_velocity(self.nu_B_hat)

        # self.get_logger().info(f"Output tau_P:\n{self.tau_P}", throttle_duration_sec=1.0)
        # self.get_logger().info(f"Output tau_D:\n{self.tau_D}", throttle_duration_sec=1.0)
        # self.get_logger().info(f"Output tau_L:\n{self.tau_L}", throttle_duration_sec=1.0)
        # self.get_logger().info(f"Output tau_\gamma:\n{self.tau_gamma}", throttle_duration_sec=1.0)
        self.get_logger().info(f"Output tau:\n{self.tau_ukf.flatten()}", throttle_duration_sec=1.0)

        

    # =========================================================
    def compute_desired_corners_pixel(self, Z_DES, pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0):
        half = TAG_SIZE / 2.0

        corners = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0]
        ], dtype=float)

        rx = np.deg2rad(pitch_deg)
        ry = np.deg2rad(yaw_deg)
        rz = np.deg2rad(roll_deg)

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
    def build_imu_measurement(self, msg):

        accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ], dtype=np.float64)

        gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ], dtype=np.float64)

        return np.hstack([accel, gyro])

    # =========================================================
    def cb_detection(self, msg):
        if len(msg.detections) == 0:
            self.detected_uv = None
            self.last_delta = None
            return

        det = msg.detections[0]
        self.detected_uv = np.array([[c.x, c.y] for c in det.corners],dtype=np.float64)

    # =========================================================
    def project_to_pixel(self, X, Y, Z):
        u = FX * X / Z + CX
        v = FY * Y / Z + CY
        return u, v
    
    # =========================================================
    def interaction_matrix_2d(self, x, y, Z):

        return np.array([
            [-1/Z,  0,  x/Z,   x*y,  -(1 + x*x),  y],
            [0,   -1/Z, y/Z, 1 + y*y,   -x*y,    -x]
        ])

    # =========================================================
    def interaction_matrix_3d(self, x, y, Z):

        return np.array([
            [-1/Z,  0,  x/Z,   x*y,  -(1 + x*x),  y],
            [0,   -1/Z, y/Z, 1 + y*y,   -x*y,    -x],
            [0,     0, -1/Z,   -y,        x,      0]
        ])

    # =========================================================
    def interaction_matrix_delta_2d(self, x, y, delta):

        return np.array([
            [-delta/bline,       0,          delta * x/bline,           x*y,      -(1 + x*x),  y],
            [0,            -delta/bline,     delta * y/bline,         1 + y*y,       -x*y,    -x]
        ])

    # =========================================================
    def interaction_matrix_delta_3d(self, x, y, delta):

        return np.array([
            [-delta/bline,       0,          delta * x/bline,           x*y,      -(1 + x*x),  y],
            [0,            -delta/bline,     delta * y/bline,         1 + y*y,       -x*y,    -x],
            [0,                  0,       (delta * delta) / bline,    delta * y,  -delta * x,  0]
        ])

    # =========================================================
    def build_interaction_matrix(self, state, deltas=None):
        state = np.asarray(state, dtype=np.float64).flatten()
        s = state[self.idx_sC]

        rows = []

        if self.use_3d_matrix_feature:
            for i in range(self.N):
                idx = 3 * i
                x = s[idx]
                y = s[idx + 1]
                Z = s[idx + 2]

                Li = self.interaction_matrix_3d(x, y, Z)
                rows.append(Li)

        else:
            for i in range(self.N):
                idx = 2 * i
                x = s[idx]
                y = s[idx + 1]
                delta = deltas[i] 

                Li = self.interaction_matrix_2d(x, y, delta)
                rows.append(Li)

        return np.vstack(rows)
    
    # =========================================================
    def compute_image_error(self, msg):
        e_pixel = []
        e_pixel_img = []
        e_norm = []
        measurement = []
        deltas = []

        pts = np.array([[p.x, p.y, p.z] for p in msg.polygon.points])
        for i in range(4):
            u, v = self.detected_uv[i]
            Z = pts[i, 2]
            if not np.isfinite(Z) or Z <= 0 or Z < 1e-4:
                return None

            x = (u - CX)/FX
            y = (v - CY)/FY
            delta =  bline / Z

            ud, vd = self.desired_pts[i]
            xd = (ud - CX)/FX
            yd = (vd - CY)/FY
            delta_des = bline / Z_DES

            deltas.append(Z)
            e_pixel_img.extend([u - ud, v - vd])

            if self.use_3d_matrix_feature: # Matrix 3x6 
                measurement.extend([x, y, Z])
                e_pixel.extend([u - ud, v - vd, Z - Z_DES])
                e_norm.extend([x - xd, y - yd, Z - Z_DES])
            
            else: # Matrix 2x6
                measurement.extend([x, y, Z])
                e_pixel.extend([u - ud, v - vd])
                e_norm.extend([x - xd, y - yd])

        measurement = np.asarray(measurement,dtype=np.float64).reshape(-1, 1)       
        e_pixel = np.asarray(e_pixel).reshape(-1, 1)
        e_norm = np.asarray(e_norm).reshape(-1, 1)
        measurement = measurement.reshape(-1, 1)
        e_pixel_img = np.asarray(e_pixel_img).reshape(-1, 1)
        deltas = np.asarray(deltas, dtype=np.float64)
        distance = np.mean(deltas)
        self.last_delta = deltas.copy()

        return distance, e_pixel, e_norm, measurement, e_pixel_img, deltas

    # =========================================================
    def fossen_acceleration(self, nu_B, tau):
        nu_B = np.asarray(nu_B, dtype=np.float64).reshape(6)
        tau = np.asarray(tau, dtype=np.float64).reshape(6)

        gamma = self.compute_gamma(nu_B.reshape(-1,1))
        nu_dot_B = self.Minv @ (tau + gamma)

        return nu_dot_B

    # =========================================================
    def build_Q_static(self, dt):
        Q = np.zeros((self.nx, self.nx), dtype=np.float64)

        q_Sc = np.tile([self.q_u, self.q_v, self.q_Z],self.N)

        Q[self.idx_sC,self.idx_sC] = np.diag(q_Sc)
        Q[self.idx_vB,self.idx_vB] = (np.eye(3) * self.q_vB)
        Q[self.idx_wB,self.idx_wB] = (np.eye(3) * self.q_wB)
        Q[self.idx_bg,self.idx_bg] = (np.eye(3) * self.sigma_bg_rw**2 * dt)
        Q[self.idx_aB,self.idx_aB] = (np.eye(3) * self.q_aB)
        Q[self.idx_ba,self.idx_ba] = (np.eye(3) * self.sigma_ba_rw**2 * dt)
        Q[self.idx_bo,self.idx_bo] = (np.eye(3) * self.sigma_bo_rw**2 * dt)

        return Q

    # =========================================================
    def build_Q(self, dt):
        Q = np.zeros((self.nx, self.nx), dtype=np.float64)

        q_Sc = np.tile([self.q_u, self.q_v, self.q_Z],self.N)

        if self.tag_lost:
            adaptive_q_vB = self.q_vB * 1e-4  # Heavily dampen velocity uncertainty growth
            adaptive_q_wB = self.q_wB * 1e-4
        else:
            adaptive_q_vB = self.q_vB
            adaptive_q_wB = self.q_wB

        Q[self.idx_sC, self.idx_sC] = np.diag(q_Sc)
        Q[self.idx_vB, self.idx_vB] = (np.eye(3) * adaptive_q_vB)
        Q[self.idx_wB, self.idx_wB] = (np.eye(3) * adaptive_q_wB)
        Q[self.idx_bg, self.idx_bg] = (np.eye(3) * self.sigma_bg_rw**2 * dt)
        Q[self.idx_aB, self.idx_aB] = (np.eye(3) * self.q_aB)
        Q[self.idx_ba, self.idx_ba] = (np.eye(3) * self.sigma_ba_rw**2 * dt)
        Q[self.idx_bo, self.idx_bo] = (np.eye(3) * self.sigma_bo_rw**2 * dt)

        return Q

    # =========================================================
    def initialize_ukf_from_camera(self, z_camera, z_imu=None):
        z_camera = np.asarray(z_camera, dtype=np.float64).reshape(self.n_ft)

        s_C0 = z_camera.copy()
        v_B0 = np.zeros(3)
        w_B0 = np.zeros(3)
        b_g0 = np.zeros(3)
        a_B0 = np.zeros(3)
        b_a0 = np.zeros(3)
        b_o0 = np.zeros(3)
        if z_imu is not None:
            z_imu = np.asarray(z_imu, dtype=np.float64).reshape(6)

        self.ukf_x = self.pack_state(s_C0, v_B0, w_B0, b_g0, a_B0, b_a0, b_o0)

        P0 = np.zeros((self.nx, self.nx), dtype=np.float64)
        if self.use_3d_matrix_feature:
            P0[self.idx_sC, self.idx_sC] = (np.eye(12) * 1e-3)
        else:
            P0[self.idx_sC, self.idx_sC] = (np.eye(8) * 1e-3)
        P0[self.idx_vB, self.idx_vB] = (np.eye(3) * 1e-2)
        P0[self.idx_wB, self.idx_wB] = (np.eye(3) * 1e-2)
        P0[self.idx_bg, self.idx_bg] = (np.eye(3) * 1e-4)
        P0[self.idx_aB, self.idx_aB] = (np.eye(3) * 1e-4)
        P0[self.idx_ba, self.idx_ba] = (np.eye(3) * 1e-2)
        P0[self.idx_bo, self.idx_bo] = (np.eye(3) * 1e-2)
        
        self.ukf_P = P0
        self.ukf_initialized = True
        self.get_logger().info("UKF initialized from stereo camera measurement.")

    # =========================================================
    def generate_sigma_points(self, x, P):
        x = np.asarray(x, dtype=np.float64).reshape(self.nx)
        P = np.asarray(P, dtype=np.float64)
        P = 0.5 * (P + P.T)
        jitter = 1e-9 * np.eye(self.nx)

        try:
            S = np.linalg.cholesky(self.gamma * (P + jitter))

        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals = np.maximum(eigvals, 1e-12)
            P_fixed = (eigvecs @ np.diag(eigvals) @ eigvecs.T)
            S = np.linalg.cholesky(self.gamma * P_fixed)

        sigma = np.zeros((self.n_sigma, self.nx), dtype=np.float64)
        sigma[0] = x

        for i in range(self.nx):
            sigma[i + 1] = (x + S[:, i])
            sigma[i + 1 + self.nx] = (x - S[:, i])

        return sigma

    # =========================================================
    def weighted_mean(self, sigma_points):
        sigma_points = np.asarray(sigma_points, dtype=np.float64)

        return np.sum(self.Wm[:, None] * sigma_points, axis=0)

    # =========================================================
    def state_covariance(self, sigma_points, x_mean):
        P = np.zeros((self.nx, self.nx), dtype=np.float64)

        for i in range(self.n_sigma):
            dx = sigma_points[i] - x_mean
            P += (self.Wc[i] * np.outer(dx, dx))

        return P

    # =========================================================
    def measurement_covariance(self, z_sigma, z_mean, R):
        z_sigma = np.asarray(z_sigma, dtype=np.float64)
        z_mean = np.asarray(z_mean, dtype=np.float64)

        measurement_dim = z_sigma.shape[1]

        S = np.zeros((measurement_dim, measurement_dim), dtype=np.float64)

        for i in range(self.n_sigma):
            dz = z_sigma[i] - z_mean
            S += self.Wc[i] * np.outer(dz, dz)

        S += R
        S = 0.5 * (S + S.T)
        S += 1e-10 * np.eye(measurement_dim)

        return S

    # =========================================================
    def cross_covariance(self, sigma_points, x_mean, z_sigma, z_mean):
        measurement_dim = z_sigma.shape[1]
        Pxz = np.zeros((self.nx, measurement_dim), dtype=np.float64)

        for i in range(self.n_sigma):
            dx = sigma_points[i] - x_mean
            dz = z_sigma[i] - z_mean
            Pxz += self.Wc[i] * np.outer(dx, dz)

        return Pxz

    # =========================================================
    def ukf_process_model(self, state, dt, tau=None):
        state = np.asarray(state, dtype=np.float64).reshape(self.nx)

        s_C, v_B, w_B, b_g, a_B, b_a, b_o = self.unpack_state(state)

        if self.tag_lost:
            # Decay velocity by 5% every step to simulate water drag
            # This absolutely prevents v_B from integrating to infinity
            v_B = v_B * 0.95 
            w_B = w_B * 0.90
            
            # Also crush the erroneous body acceleration 
            a_B = a_B * 0.50

        nu_B = np.concatenate([v_B, w_B])
        nu_C = self.b_to_c_velocity(nu_B)

        # State is [u, v, Z], so L must produce
        # [u_dot, v_dot, Z_dot].
        if self.use_3d_matrix_feature:
            L_sigma = self.build_interaction_matrix(state)
        
        else:
            L_sigma = self.build_interaction_matrix(s_C, self.last_delta)

        sC_next = s_C + dt * (L_sigma @ nu_C)
        vB_next = v_B + dt * a_B
        wB_next = w_B + dt * b_o

        bg_next = b_g
        aB_next = a_B
        ba_next = b_a
        bo_next = b_o

        return self.pack_state(sC_next, vB_next, wB_next, bg_next, aB_next, ba_next, bo_next)

    # =========================================================
    def ukf_predict(self, x, P, dt, tau):
        sigma = self.generate_sigma_points(x, P)
        sigma_pred = np.zeros_like(sigma)

        for i in range(self.n_sigma):
            sigma_pred[i] = (self.ukf_process_model(sigma[i], dt, tau))

        x_pred = self.weighted_mean(sigma_pred)
        P_pred = self.state_covariance(sigma_pred, x_pred)

        P_pred += self.build_Q(dt)
        P_pred = 0.5 * (P_pred + P_pred.T)
        P_pred += (1e-10 * np.eye(self.nx))

        return (x_pred, P_pred, sigma_pred)

    # =========================================================
    def camera_measurement_model(self, state):
        state = np.asarray(state, dtype=np.float64).reshape(self.nx)

        return state[self.idx_sC].copy()

    def camera_measurement_model_noise(self, z_camera):
        z_camera = np.asarray(z_camera, dtype=np.float64).reshape(self.n_ft)

        if not self.enable_camera_noise:
            return z_camera.copy()

        sigma_x = float(0.334788) / FX
        sigma_y = float(0.363501) / FY
        sigma_Z = float(0.01)
        noise = np.zeros(self.n_ft, dtype=np.float64)

        for i in range(self.N):
            idx = 3 * i
            noise[idx] = self.camera_noise_rng.normal(0.0, sigma_x)
            noise[idx + 1] = self.camera_noise_rng.normal(0.0, sigma_y)
            noise[idx + 2] = self.camera_noise_rng.normal(0.0, sigma_Z)

        return z_camera + noise

    # =========================================================
    def camera_measurement_model_pinhole(self, state):
        state = np.asarray(state, dtype=np.float64).reshape(self.nx)

        s = state[self.idx_sC]

        measurement = []

        for i in range(self.N):
            idx = 3 * i

            X = s[idx]
            Y = s[idx + 1]
            Z = s[idx + 2]

            if not np.isfinite(Z) or Z <= 1e-6:
                raise ValueError(f"Invalid depth Z[{i}] = {Z}")

            u, v = self.project_to_pixel(X, Y, Z)
            u = FX * X / Z + CX
            v = FY * Y / Z + CY

            measurement.extend([u, v])

        return np.asarray(measurement, dtype=np.float64)

    # =========================================================
    def predict_camera_sigma_points(self, sigma_points):
        z_sigma = np.zeros((self.n_sigma, self.n_ft), dtype=np.float64)

        for i in range(self.n_sigma):
            z_sigma[i] = self.camera_measurement_model(sigma_points[i])

        return z_sigma

    # =========================================================
    def ukf_update_camera(self, x_pred, P_pred, sigma_pred, z_camera):
        z_camera = np.asarray(z_camera, dtype=np.float64).reshape(self.n_ft)

        z_sigma = self.predict_camera_sigma_points(sigma_pred)
        z_mean = self.weighted_mean(z_sigma)

        S = self.measurement_covariance(z_sigma, z_mean, self.R_camera)
        Pxz = self.cross_covariance(sigma_pred, x_pred, z_sigma, z_mean)

        K = np.linalg.solve(S, Pxz.T).T
        innovation = z_camera - z_mean
        x_update = x_pred + K @ innovation
        P_update = P_pred - K @ S @ K.T
        P_update = 0.5 * (P_update + P_update.T)
        P_update += 1e-10 * np.eye(self.nx)

        return (x_update, P_update, innovation, S, K, z_mean)

    # =========================================================
    def imu_measurement_model(self, state, R_NB):
        state = np.asarray(state, dtype=np.float64).reshape(self.nx)

        s_C, v_B, w_B, b_g, a_B, b_a, b_o = self.unpack_state (state)
        
        g_N = np.array([0.0, 0.0, 9.80665],dtype=np.float64)
        g_B = R_NB.T @ g_N

        a_pred_B = (a_B + np.cross(w_B, v_B) - g_B + b_a)
        omega_pred_B = (w_B + b_g)

        z_pred = np.concatenate([a_pred_B, omega_pred_B])

        return z_pred

    # =========================================================
    def predict_imu_sigma_points(self, sigma_points, R_NB):
        z_sigma = np.zeros((self.n_sigma, 6), dtype=np.float64)

        for i in range(self.n_sigma):
            z_sigma[i] = (self.imu_measurement_model( sigma_points[i], R_NB))

        return z_sigma
    
    # =========================================================
    def ukf_update_imu_static(self, x_pred, P_pred, sigma_pred, z_imu, R_NB):
        z_imu = np.asarray(z_imu, dtype=np.float64).reshape(6)
        z_sigma = self.predict_imu_sigma_points(sigma_pred, R_NB)
        z_mean = self.weighted_mean(z_sigma)

        S = self.measurement_covariance(z_sigma, z_mean, R_imu)
        Pxz = self.cross_covariance(sigma_pred, x_pred, z_sigma, z_mean)
        K = np.linalg.solve(S, Pxz.T).T

        innovation = z_imu - z_mean
        x_update = x_pred + K @ innovation
        P_update = P_pred - K @ S @ K.T
        P_update = 0.5 * (P_update + P_update.T)
        P_update += 1e-10 * np.eye(self.nx)

        return x_update, P_update, innovation, S, K, z_mean

    # =========================================================
    def ukf_update_imu(self, x_pred, P_pred, sigma_pred, z_imu, R_NB):
        z_imu = np.asarray(z_imu, dtype=np.float64).reshape(6)
        z_sigma = self.predict_imu_sigma_points(sigma_pred, R_NB)
        z_mean = self.weighted_mean(z_sigma)

        innovation = z_imu - z_mean
        S_standard = self.measurement_covariance(z_sigma, z_mean, R_imu)

        mahalanobis_dist = innovation.T @ np.linalg.inv(S_standard) @ innovation

        gamma = 12.59

        if mahalanobis_dist > gamma:
            # The measurement is highly unlikely. Scale up R to suppress the Kalman Gain.
            scale_factor = mahalanobis_dist / gamma
            
            # Recompute S with the heavily penalized R matrix
            S = self.measurement_covariance(z_sigma, z_mean, R_imu * scale_factor)
            
            self.get_logger().warn(
                f"Adaptive R Triggered! Scale Factor: {scale_factor:.2f}",
                throttle_duration_sec=1.0
            )
        else:
            S = S_standard

        Pxz = self.cross_covariance(sigma_pred, x_pred, z_sigma, z_mean)
        K = np.linalg.solve(S, Pxz.T).T

        x_update = x_pred + K @ innovation
        P_update = P_pred - K @ S @ K.T
        P_update = 0.5 * (P_update + P_update.T)
        P_update += 1e-10 * np.eye(self.nx)

        return x_update, P_update, innovation, S, K, z_mean
    
    # =========================================================
    def imu_flu_to_body(self, accel_flu, gyro_flu):
        accel_flu  = np.asarray(accel_flu , dtype=np.float64).reshape(3)
        gyro_flu = np.asarray(gyro_flu, dtype=np.float64).reshape(3)

        accel_B = R_IB @ accel_flu 
        gyro_B = R_IB @ gyro_flu

        return accel_B, gyro_B

    # =========================================================
    def compute_dt(self, timestamp):
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            return 0.0

        dt = timestamp - self.last_timestamp

        if dt < 0.0:
            raise ValueError(f"Out-of-order measurement: dt={dt:.6f}")

        self.last_timestamp = timestamp

        return dt

    # =========================================================
    def get_dt(self):
        now = self.get_clock().now()

        if self.last_time is None:
            dt = 0.04
        else:
            dt = (now - self.last_time).nanoseconds * 1e-9

        self.last_time = now
        return dt

    # =========================================================
    def update_L_hat(self):
        if getattr(self, 'tag_lost', False) and self.L_hat is not None:
            return self.L_hat
        
        feature_hat = self.ukf_x[:self.feature_dim].flatten()

        if self.use_3d_matrix_feature:
            self.L_hat = self.build_interaction_matrix(feature_hat)

        else:
            if self.last_delta is None:
                self.L_hat = None
                return

            self.L_hat = self.build_interaction_matrix(feature_hat, self.last_delta)

        return self.L_hat

    # =========================================================
    def compute_Ldot(self, dt):
        if self.L_hat is None:
            return

        if dt <= 1e-6:
            self.L_dot = (np.zeros_like(self.L_hat)
                if self.L_hat is not None
                else None)
            return

        if self.L_prev is None:
            self.L_prev = self.L_hat.copy()
            self.L_dot = np.zeros_like(self.L_hat)
            return

        raw_L_dot = (self.L_hat - self.L_prev) / dt
        
        alpha = 0.15  

        if hasattr(self, 'L_dot') and self.L_dot is not None:
            self.L_dot = alpha * raw_L_dot + (1.0 - alpha) * self.L_dot
        else:
            self.L_dot = raw_L_dot

        self.L_prev = self.L_hat.copy()

        self.L_dot = (self.L_hat - self.L_prev) / dt
        self.L_prev = self.L_hat.copy()

    # =========================================================
    def interaction_matrix_dot(self, x, y, delta, x_dot, y_dot, delta_dot):
        b = bline

        Lx = np.array([
            [0, 0, delta/b, y, -2*x, 0],
            [0, 0, 0,     0, -y,   -1]])

        Ly = np.array([
            [0, 0, 0,       x, 0, 1],
            [0, 0, delta/b, 2*y, -x, 0]])

        Ldelta = np.array([
            [-1/b, 0, x/b, 0, 0, 0],
            [0, -1/b, y/b, 0, 0, 0]])

        return ( Lx * x_dot
                + Ly * y_dot
                + Ldelta * delta_dot)

    # =========================================================
    def compute_Ldot_analytical(self):
        if self.L_hat is None: 
            self.L_dot = None 
            return None 

        if self.nu_B_hat is None: 
            self.L_dot = None 
            return None
        
        feature_hat = self.ukf_x[:self.feature_dim].flatten()

        if self.use_3d_matrix_feature: 
            Ldot_rows = [] 

            for i in range(self.N): 
                idx = 3 * i 
                x = feature_hat[idx] 
                y = feature_hat[idx + 1] 
                delta = feature_hat[idx + 2] 

                feature_dot = self.L_hat @ self.T_bc @ self.nu_B_hat.flatten()
                
                x_dot = feature_dot[idx] 
                y_dot = feature_dot[idx + 1]
                delta_dot = feature_dot[idx + 2] 

                Ldot_i = self.interaction_matrix_dot(x, y, delta, x_dot, y_dot, delta_dot ) 
                Ldot_rows.append(Ldot_i) 

            self.L_dot = np.vstack(Ldot_rows)

        else: 
            if self.last_delta is None: 
                self.L_dot = None 
                return None 

            Ldot_rows = [] 

            for i in range(self.N): 
                idx = 2 * i 
                x = feature_hat[idx] 
                y = feature_hat[idx + 1] 
                delta = self.last_delta[i]

                feature_dot = self.L_hat @ self.T_bc @ self.nu_B_hat.flatten()
                x_dot = feature_dot[idx]
                y_dot = feature_dot[idx + 1]  

                Li_3d = self.interaction_matrix_feature_3d( x, y, delta ) 
                delta_dot = ( Li_3d[2, :] @ self.T_bc @ self.nu_B_hat.flatten())
                Ldot_i = self.interaction_matrix_dot(x, y, delta, x_dot, y_dot, delta_dot ) 
                Ldot_rows.append(Ldot_i) 

            self.L_dot = np.vstack(Ldot_rows)

        return self.L_dot

    # =========================================================
    def skew(self, p):

        return np.array([
            [0,-p[2],p[1]],
            [p[2],0,-p[0]],
            [-p[1],p[0],0]
        ])

    # =========================================================
    def camera_body_adjoint(self):
        S = self.skew(P_BC_0)
        T_bc = np.block([
            [R_CB,           -S @ R_CB],
            [np.zeros((3,3)),     R_CB]])

        return T_bc

    # =========================================================
    def b_to_c_velocity(self, nu_B):
        S = self.skew(P_BC_0)

        T_CB = np.block([
            [R_CB,            -R_CB @ S],
            [np.zeros((3, 3)),     R_CB]])
        
        return T_CB @ nu_B

    # =========================================================
    def quaternion_to_rotation(self, q):
        x = q.x
        y = q.y
        z = q.z
        w = q.w

        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),   1 - 2*(x*x + z*z),   2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]], dtype=np.float64)

        return R
    
    # =========================================================
    def quaternion_to_euler(self, q):
        x = q.x
        y = q.y
        z = q.z
        w = q.w

        # Roll
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)

        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch
        sinp = 2.0 * (w * y - z * x)

        if abs(sinp) >= 1.0:
            pitch = np.copysign(np.pi / 2.0, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    # =========================================================
    def compute_damping(self, nu):
        nu = nu.flatten()
        linear = Dlin @ nu
        quadratic = Dquad @ (np.abs(nu) * nu)

        return (linear + quadratic).reshape(-1,1)

    def compute_coriolis(self, nu):
        u,v,w,p,q,r = nu.flatten()
        C = np.array([
            [0,0,0,   0,  m*w, -m*v],
            [0,0,0, -m*w,  0,   m*u],
            [0,0,0, m*v, -m*u,    0],

            [0,   m*w, -m*v,   0,    Izz*r,  -Iyy*q],
            [-m*w, 0,   m*u, -Izz*r,   0,     Ixx*p],
            [m*v, -m*u,  0,   Iyy*q, -Ixx*p,      0]])

        return C @ nu

    def compute_restoring(self):

        return np.zeros((6,1))

    # =========================================================
    def compute_gamma(self, nu):
        gamma = (self.compute_coriolis(nu) +
                self.compute_damping(nu) +
                self.compute_restoring())

        return gamma

    # =========================================================
    def compute_alpha(self, L):

        return L @ self.T_bc @ self.Minv

    # =========================================================
    def compute_control_tau(self, distance, e_norm, e_pixel, dt):
        alpha_current = self.compute_alpha(self.L_hat)
        gamma = self.compute_gamma(self.nu_B_hat.reshape(-1,1))

        if self.freeze_L_alpha:
            if not self.freeze_initialized:
                self.L_frozen = self.L_hat.copy()
                self.alpha_frozen = alpha_current.copy()

                # Compute pseudoinverse only once
                self.alpha_pinv_frozen = np.linalg.pinv(self.alpha_frozen)

                self.freeze_initialized = True

                self.get_logger().warn("================================================")
                self.get_logger().warn("FREEZING L AND ALPHA AT FIRST VALID ITERATION")
                self.get_logger().warn(f"L shape     : {self.L_frozen.shape}")
                self.get_logger().warn(f"alpha shape : {self.alpha_frozen.shape}")
                self.get_logger().warn(f"rank(alpha) : "
                                f"{np.linalg.matrix_rank(self.alpha_frozen)}")
                self.get_logger().warn("================================================")
            
            L = self.L_frozen.copy()
            alpha = self.alpha_frozen.copy()
            alpha_pinv = self.alpha_pinv_frozen

            np.set_printoptions(
                precision=8,
                suppress=False,
                linewidth=200
            )

            self.get_logger().warn(f"L0 =\n{self.L_frozen}")
            self.get_logger().warn(f"alpha0 =\n{self.alpha_frozen}")

            U, S, Vt = np.linalg.svd(self.alpha_frozen, full_matrices=False)

            sigma_min = S[-1]
            sigma_max = S[0]

            cond_alpha = (sigma_max /max(sigma_min, 1e-12))

            L_change = (np.linalg.norm(self.L_hat - self.L_frozen,ord='fro')/
                        max(np.linalg.norm(self.L_frozen,ord='fro'),1e-12))

            alpha_change = (np.linalg.norm(alpha_current - self.alpha_frozen,ord='fro')/
                        max(np.linalg.norm(self.alpha_frozen,ord='fro'),1e-12))

            for i in range(6):
                self.get_logger().warn(
                    f"mode {i+1}: "
                    f"sigma={S[i]:.6e}, "
                    f"V={Vt[i]}"
                )

        else:

            # Normal controller
            L = self.L_hat
            alpha = alpha_current
            alpha_pinv = np.linalg.pinv(alpha)

        e_dot_hat = self.L_hat @ self.T_bc @ self.nu_B_hat.reshape(-1,1)
        l_dot = self.L_dot @ self.T_bc @ self.nu_B_hat.reshape(-1,1)

        tau_P = - self.Kp @ alpha_pinv @ e_norm
        # tau_D = - self.Kd @ alpha_pinv @ e_dot_hat
        # tau_L = - alpha_pinv @ l_dot
        # tau_gamma = alpha_pinv @ alpha @ gamma

        # tau = tau_P + tau_D + tau_L + tau_gamma
        tau = tau_P.copy()

        if np.max(np.abs(e_pixel)) < dead_band:
            tau[:] = 0

        self.limit_force(tau)
        self.tau_P = tau_P
        # self.tau_D = tau_D
        # self.tau_L = tau_L
        # self.tau_gamma = tau_gamma

        self.get_logger().info(
            f"L_change={L_change:.6e}, "
            f"alpha_change={alpha_change:.6e}"
        )
        self.get_logger().warn(f"Frozen alpha singular values = {S}")
        self.get_logger().warn(f"Frozen sigma_min = {sigma_min:.6e}")
        self.get_logger().warn(f"Frozen sigma_max = {sigma_max:.6e}")
        self.get_logger().warn(f"Frozen condition number = {cond_alpha:.6e}")

        return tau.flatten()

    # =========================================================
    def compute_control_tau_debug(self, distance, e_norm, e_pixel, dt):
        self.update_L_hat()
        self.compute_Ldot(dt)

        if self.L_hat is None or self.L_dot is None:
            return np.zeros(6)

        alpha = self.compute_alpha(self.L_hat)
        gamma = self.compute_gamma(self.nu_B_hat.reshape(-1,1))

        e_dot_hat = self.L_hat @ self.T_bc @ self.nu_B_hat.reshape(-1,1)
        l_dot = self.L_dot @ self.T_bc @ self.nu_B_hat.reshape(-1,1)

        alpha_inv = np.linalg.pinv(alpha)

        tau_P = - self.Kp @ alpha_inv @ e_norm
        tau_D = - self.Kd @ alpha_inv @ e_dot_hat
        tau_L = - alpha_inv @ l_dot
        tau_gamma = alpha_inv @ alpha @ gamma

        tau = tau_P + tau_D + tau_L + tau_gamma

        if np.max(np.abs(e_pixel)) < dead_band:
            tau[:] = 0

        self.limit_force(tau)
        self.tau_P = tau_P
        self.tau_D = tau_D
        self.tau_L = tau_L
        self.tau_gamma = tau_gamma

        return tau.flatten()

    # =========================================================
    def compute_control_tau_DLS(self, distance, e_norm, e_pixel, dt):
        self.update_L_hat()
        self.compute_Ldot(dt)

        if self.L_hat is None or self.L_dot is None:
            return np.zeros(6)

        alpha = self.compute_alpha(self.L_hat)
        gamma = self.compute_gamma(self.nu_B_hat.reshape(-1,1))

        A = alpha.T @ alpha + mu**2 * np.eye(6)

        e_dot_hat = self.L_hat @ self.T_bc @ self.nu_B_hat.reshape(-1,1)
        l_dot = self.L_dot @ self.T_bc @ self.nu_B_hat.reshape(-1,1)
        gams = alpha @ gamma

        tau_P = - self.Kp @ np.linalg.solve(A, alpha.T @ e_norm).flatten()
        tau_D = - self.Kd @ np.linalg.solve(A, alpha.T @ e_dot_hat).flatten()
        tau_L = - np.linalg.solve(A, alpha.T @ l_dot).flatten()
        tau_gamma = np.linalg.solve(A, alpha.T @ gams).flatten()

        tau = tau_P + tau_D + tau_L + tau_gamma

        if np.max(np.abs(e_pixel)) < dead_band:
            tau[:] = 0

        self.limit_force(tau)
        self.tau_P = tau_P
        self.tau_D = tau_D
        self.tau_L = tau_L
        self.tau_gamma = tau_gamma

        return tau.flatten()
    
    # =========================================================
    def compute_control_tau1(self, L_hat, distance, e_norm, e_pixel, dt):
        self.update_L_hat()
        self.compute_Ldot(dt)

        if self.L_hat is None or self.L_dot is None:
            return np.zeros(6)

        alpha = self.compute_alpha(self.L_hat)
        gamma = self.compute_gamma(self.nu_B_hat.reshape(-1,1))

        rhs = lambda_gain * e_norm + self.L_dot @ self.T_bc @ self.nu_B_hat - alpha @ gamma
        tau = -np.linalg.pinv(alpha) @ rhs

        if np.max(np.abs(e_pixel)) < dead_band:
            tau[:] = 0

        self.limit_force(tau)
        tau = tau.flatten()

        return tau

    # =========================================================
    def compute_control_ibvs1(self, L, distance, e_norm, e_pixel, dt):
        A = L.T @ L + mu**2 * np.eye(6)

        if self.e_integral is None:
            self.e_integral = np.zeros_like(e_norm)

        self.e_integral += e_norm * dt

        vp = np.linalg.solve(A, L.T @ e_norm).flatten()
        vi = np.linalg.solve(A, L.T @ self.e_integral).flatten()

        Vc = -lambda_gain * (self.Kp @ vp + self.Ki @ vi)

        if np.max(np.abs(e_pixel)) < dead_band:
            Vc[:] = 0

        self.limit_velocity(Vc)

        return Vc

    # =========================================================
    def compute_control_ibvs2(self, L, distance, e_norm, pixel_norm):
        L_xy = L[:,[0,1,3,4]]
        L_z  = L[:,[2,5]]

        p1 = self.detected_uv[0]
        p2 = self.detected_uv[1]
        measured_roll = np.arctan2(
            p2[1] - p1[1],
            p2[0] - p1[0]
        )

        depth_error = bline/distance - bline/Z_DES
        roll_error = np.arctan2(np.sin(measured_roll-self.desired_roll), 
                                np.cos(measured_roll-self.desired_roll))

        vz = -self.kz * depth_error
        wz = -self.kw * roll_error

        A = L_xy.T @ L_xy + mu**2*np.eye(4)
        b_p = L_xy.T @ e_norm
        v_p = np.linalg.solve(A, b_p).flatten()

        rhs = self.lambda_gain * e_norm + L_z @ np.array([[vz],[wz]])

        b = L_xy.T @ rhs
        Vxy = - np.linalg.solve(A,b)

        vx = Vxy[0,0]
        vy = Vxy[1,0]
        wx = Vxy[2,0]
        wy = Vxy[3,0]


        Vc = np.array([vx, vy, vz,
                    wx, wy, wz]).reshape(6,1)

        if np.max(np.abs(pixel_norm)) < dead_band:
            Vc[:] = 0

    # =========================================================
    def compute_control_ibvs3(self, L, distance, e_norm, pixel_norm):
        _ = distance

    # =========================================================
    def compute_control_ibvs4(self, L, distance, e_norm, pixel_norm):
        _ = distance

    # =========================================================
    def limit_velocity(self, Vc):
        vnorm = np.linalg.norm(Vc[:3])
        if vnorm > MAX_LIN_VEL:
            Vc[:3] *= MAX_LIN_VEL / vnorm

        wnorm = np.linalg.norm(Vc[3:])
        if wnorm > MAX_ANG_VEL:
            Vc[3:] *= MAX_ANG_VEL / wnorm

    # =========================================================
    def limit_force(self, tau):
        force = np.linalg.norm(tau[:3])
        if force > MAX_FORCE:
            tau[:3] *= MAX_FORCE / force

        momen = np.linalg.norm(tau[3:])
        if momen > MAX_MOMEN:
            tau[3:] *= MAX_MOMEN / momen

    # =========================================================
    def camera_to_body(self, Vc):
        v = Vc[:3]
        w = Vc[3:]

        Wb = R_CB @ w
        Vb = R_CB @ v + np.cross(Wb, P_BC_0.reshape(3))

        Vb[0] = Vb[0]
        Vb[1] = Vb[1]
        Vb[2] = Vb[2]
        Wb[0] = Wb[0]
        Wb[1] = Wb[1]
        Wb[2] = Wb[2]

        return Vb, Wb

    # =========================================================
    def publish_twist(self, pub, frame, stamp, linear, angular):
        msg = TwistStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame

        msg.twist.linear.x = float(linear[0])
        msg.twist.linear.y = float(linear[1])
        msg.twist.linear.z = float(linear[2])
        msg.twist.angular.x = float(angular[0])
        msg.twist.angular.y = float(angular[1])
        msg.twist.angular.z = float(angular[2])

        pub.publish(msg)

    # =========================================================
    def vel_to_pwm(self, v, bias=0):
        return int(np.clip(1500 + 400 * v + bias, 1100, 1900))

    # =========================================================
    def compute_vel_pwm(self, Vb, Wb):

        pwm = [1500] * 18
        pwm[4] = self.vel_to_pwm(Vb[0])
        pwm[5] = self.vel_to_pwm(Vb[1])
        pwm[2] = self.vel_to_pwm(Vb[2], self.HEAVE_BIAS)
        pwm[1] = self.vel_to_pwm(Wb[0])
        pwm[0] = self.vel_to_pwm(Wb[1])
        pwm[3] = self.vel_to_pwm(Wb[2])
        self.current_pwm = pwm

        return pwm

    # =========================================================
    def publish_torque(self, pub, frame, stamp, tau):
        msg = WrenchStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame

        tau = tau.flatten()

        msg.wrench.force.x = float(tau[0])
        msg.wrench.force.y = float(tau[1])
        msg.wrench.force.z = float(tau[2])
        msg.wrench.torque.x = float(tau[3])
        msg.wrench.torque.y = float(tau[4])
        msg.wrench.torque.z = float(tau[5])

        pub.publish(msg)

    # =========================================================
    def force_to_pwm(self, F, bias=0):
        return int(np.clip(1500 + F * 25 + bias, 1100, 1900))

    # =========================================================
    def compute_force_pwm(self, tau):

        pwm = [1500] * 18
        pwm[4] = self.force_to_pwm(tau[0])
        pwm[5] = self.force_to_pwm(tau[1])
        pwm[2] = self.force_to_pwm(tau[2], self.HEAVE_BIAS)
        pwm[1] = self.force_to_pwm(tau[3])
        pwm[0] = self.force_to_pwm(tau[4])
        pwm[3] = self.force_to_pwm(tau[5])
        self.current_pwm = pwm

        return pwm

    # =========================================================
    def ukf_logging(self, source, innovation, K, z, x_prior=None):
        innovation = np.asarray(innovation, dtype=np.float64).reshape(-1)
        K = np.asarray(K, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64).reshape(-1)

        if source == "imu":
            source_id = 0.0

        elif source == "camera":
            source_id = 1.0

        else:
            source_id = -1.0

        innovation_norm = np.linalg.norm(innovation)
        gain_norm = np.linalg.norm(K, ord="fro")

        state_correction = K @ innovation
        state_correction_norm = np.linalg.norm(state_correction)

        measurement_norm = np.linalg.norm(z)

        # idx_nuB = np.concatenate([
        #     np.arange(self.idx_vB.start, self.idx_vB.stop),
        #     np.arange(self.idx_wB.start, self.idx_wB.stop)
        # ])

        # P_nu = self.ukf_P[np.ix_(idx_nuB, idx_nuB)]

        P_nu = self.ukf_P[np.ix_(self.idx_nuB, self.idx_nuB)]
        P_nu = 0.5 * (P_nu + P_nu.T)

        sigma_nu = np.sqrt(np.maximum(np.diag(P_nu), 0.0))

        nu_B = np.concatenate([self.ukf_x[self.idx_vB], self.ukf_x[self.idx_wB]])

        msg = Float32MultiArray()
        msg.data = [
            float(source_id),
            float(innovation_norm),
            float(gain_norm),
            float(state_correction_norm),
            float(measurement_norm),

            *sigma_nu.tolist(),
            *nu_B.tolist(),
        ]

        self.ukf_data_pub.publish(msg)

        self.get_logger().info(
            f"[UKF-{source.upper()}] "
            f"innov={innovation_norm:.5e} | "
            f"||K||={gain_norm:.5e} | "
            f"||K*innov||={state_correction_norm:.5e} | ",
            throttle_duration_sec=1.0
        )

    # =========================================================
    # def log_debug(self, Vc=None, Vb=None, Wb=None, tau=None, pwm=None):
    def log_debug(self, tau=None, pwm=None):
        # if Vc is not None:
        #     self.get_logger().info(
        #         f"Cam_LatX = {Vc[0]:.3f} |"
        #         f"Cam_LatY = {Vc[1]:.3f} |"
        #         f"Cam_LatZ = {Vc[2]:.3f} |"
        #         f"Cam_RotX = {Vc[3]:.3f} |"
        #         f"Cam_RotY = {Vc[4]:.3f} |"
        #         f"Cam_RotZ = {Vc[5]:.3f} |",
        #         throttle_duration_sec=0.5)
            
        # if Vb is not None:
        #     self.get_logger().info(
        #         f"Surge = {Vb[0]:.3f} |"
        #         f"Sway = {Vb[1]:.3f} |"
        #         f"Heave = {Vb[2]:.3f} |"
        #         f"Roll = {Wb[0]:.3f} |"
        #         f"Pitch = {Wb[1]:.3f} |"
        #         f"Yaw = {Wb[2]:.3f} ",
        #         throttle_duration_sec=0.5)
            
        if tau is not None:
            self.get_logger().info(
                f"Surge = {tau[0]:.3f} |"
                f"Sway = {tau[1]:.3f} |"
                f"Heave = {tau[2]:.3f} |"
                f"Roll = {tau[3]:.3f} |"
                f"Pitch = {tau[4]:.3f} |"
                f"Yaw = {tau[5]:.3f} ",
                throttle_duration_sec=1.0)
        
        if pwm is not None:
            self.get_logger().info(
                f"Surge={pwm[4]} "
                f"Sway={pwm[5]} "
                f"Heave={pwm[2]} "
                f"Roll={pwm[1]} "
                f"Pitch={pwm[0]} "
                f"Yaw={pwm[3]}",
                throttle_duration_sec=1.0)

    def log_covariance_blocks(self, P):

        blocks = {
            "sC": self.idx_sC,
            "vB": self.idx_vB,
            "wB": self.idx_wB,
            "bg": self.idx_bg,
            "aB": self.idx_aB,
            "ba": self.idx_ba,
            "bo": self.idx_bo,
        }

        for name, idx in blocks.items():

            P_block = P[np.ix_(
                np.arange(idx.start, idx.stop),
                np.arange(idx.start, idx.stop)
            )]

            P_block = 0.5 * (P_block + P_block.T)

            self.get_logger().info(
                f"P[{name}] "
                f"trace={np.trace(P_block):.3e}, "
                f"max={np.max(np.abs(P_block)):.3e}"
            )

    # =========================================================
    def publish_rc(self):
        rc_msg = OverrideRCIn()
        # rc_msg.channels = [int(c) for c in self.current_pwm]
        rc_msg.channels = list(map(int, self.current_pwm))
        self.rc_override_pub.publish(rc_msg)

    # =========================================================
    def publish_error(self, e_pixel, e_norm):
        err_px_msg = Float32MultiArray()
        # err_px_msg.data = np.array(e_pixel, dtype=np.float32).flatten().tolist()
        err_px_msg.data = e_pixel.astype(np.float32).ravel().tolist()
        self.err_px_pub.publish(err_px_msg)

        err_no_msg = Float32MultiArray()
        # err_no_msg.data = np.array(e_norm, dtype=np.float32).flatten().tolist()
        err_no_msg.data = e_norm.astype(np.float32).ravel().tolist()
        self.err_no_pub.publish(err_no_msg)
        
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
            
            # Only reset to neutral IF the tag is actually lost
            self.current_pwm = [1500] * 18
            self.current_pwm[2] = 1500 + 100      # RC3 (Heave), +200 bias
            self.current_pwm[5] = 1500 - 30

# =============================================================
def main():
    rclpy.init()
    node = IBVSRCController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()