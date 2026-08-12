## Stereo-enhanced IBVS Ls 3x6 all normalized

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import numpy as np

from apriltag_msgs.msg import AprilTagDetectionArray
from geometry_msgs.msg import PolygonStamped, TwistStamped, WrenchStamped
from std_msgs.msg import Float32MultiArray, Int16MultiArray
from mavros_msgs.msg import OverrideRCIn
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Imu
from cv_bridge import CvBridge

from ibvs.constants import *

class IBVSRCController(Node):
    def __init__(self):
        super().__init__("ibvs_rc_controller")

        self.bridge = CvBridge()

        # ---------------- Subscribers ----------------
        self.sub_corners = self.create_subscription(PolygonStamped, "/apriltag/corners", self.cb_corners, qos_profile_sensor_data)
        self.sub_detection = self.create_subscription(AprilTagDetectionArray, "/detection1", self.cb_detection, 10)
        self.camera_imu_sub = self.create_subscription(Imu, '/camera/imu', self.cb_camera_imu, 50)
        self.fcu_imu_sub = self.create_subscription(Imu, '/mavros/imu/data', self.cb_fcu_imu, 50)

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


        # self.compute_control = self.compute_control_ibvs1
        self.freeze_L_alpha = True

        self.L_frozen = None
        self.alpha_frozen = None
        self.alpha_pinv_frozen = None

        self.freeze_initialized = False


        self.m = 29.0
        self.Ixx = 0.492558 + 0.16
        self.Iyy = 0.758506 + 0.30
        self.Izz = 0.919455 + 0.30

        self.Dlin = np.diag([4.0, 4.0, 5.0,
                        0.0, 0.0, 0.8])

        self.Dquad = np.diag([6.0, 6.0, 8.0,
                        0.4, 1.19, 0.482])

        self.M = np.diag([self.m, self.m, self.m,
                        self.Ixx, self.Iyy, self.Izz])
        self.Minv = np.linalg.inv(self.M)
        self.R_BN = np.eye(3)
        self.R_imu = np.diag([
            1.27215**2,
            1.27215**2,
            1.27215**2,
            0.00114789**2,
            0.00114789**2,
            0.00114789**2
        ])

        self.current_pwm = [1500] * 18
        self.depth_img = None
        self.detected_uv = None
        self.last_tag_time = None
        self.last_imu_time = None
        self.last_camera_time = None
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
        self.nx = self.n_ft + 6 + 6

        self.idx_s = slice(0, self.n_ft)
        self.idx_nu = slice(self.n_ft, self.n_ft + 6)
        self.idx_ba = slice(self.n_ft + 6, self.n_ft + 9)
        self.idx_bg = slice(self.n_ft + 9, self.n_ft + 12)

        # ---------- UKF Parameters ----------
        self.alpha = 0.01
        self.beta = 2.0
        self.kappa = 0.0

        self.betaQ_feature = 0.01
        self.betaQ_velocity = 0.05
        self.betaR = 0.05
        
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

        # ---------- State ----------
        Q = np.zeros((self.nx, self.nx))
        Q[:self.feature_dim, :self.feature_dim] = np.eye(self.feature_dim) * 1e-6 # feature process noise
        Q[self.feature_dim:, self.feature_dim:] = np.eye(6) * 1e-3 # velocity process noise

        self.q_feature = 1e-6
        self.q_velocity = 1e-3

        self.sigma_ba_rw = 0.003597694747410243
        self.sigma_bg_rw = 7.692845772051322e-7
        self.camera_imu_timeshift = 0.00702

        self.ukf_x = np.zeros((self.nx,1))
        self.ukf_P = np.eye(self.nx)*1e-3
        self.ukf_Q = Q
        self.ukf_R = np.eye(self.n_ft)*1e-4

        self.tau_ukf = np.zeros((6,1))
        self.eta_ukf = np.zeros((6,1))

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
    def pack_state(self, s, nu_B, b_a, b_g): 
        state = np.zeros(self.nx, dtype=np.float64)

        state[self.idx_s] = np.asarray(s).reshape(-1)
        state[self.idx_nu] = np.asarray(nu_B).reshape(-1)
        state[self.idx_ba] = np.asarray(b_a).reshape(-1)
        state[self.idx_bg] = np.asarray(b_g).reshape(-1)

        return state
    
    # =========================================================
    def unpack_state(self, state):
        state = np.asarray(state, dtype=np.float64).reshape(-1)

        s = state[self.idx_s]
        nu_B = state[self.idx_nu]
        b_a = state[self.idx_ba]
        b_g = state[self.idx_bg]

        return s, nu_B, b_a, b_g

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
        t = self.stamp_to_sec(msg.header.stamp)

        if self.last_imu_time is None:
            self.last_imu_time = t
            self.R_BN = self.quaternion_to_rotation(msg.orientation)

            return

        dt = t - self.last_imu_time
        self.last_imu_time = t

        if dt <= 0.0 or dt > 0.1:
            return

        self.R_BN = self.quaternion_to_rotation(msg.orientation)

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

        x_pred, P_pred, sigma_pred = self.ukf_predict(self.ukf_x, self.ukf_P, dt, self.tau_ukf)
        self.ukf_x, self.ukf_P, imu_innovation, S_imu, K_imu = self.ukf_update_imu(x_pred, P_pred, sigma_pred, z_imu, self.tau_ukf, self.R_BN)

        self.s_hat = self.ukf_x[self.idx_s].copy()
        self.nu_B_hat = self.ukf_x[self.idx_nu].copy()
        self.b_a_hat = self.ukf_x[self.idx_ba].copy()
        self.b_g_hat = self.ukf_x[self.idx_bg].copy()

        self.nu_C_hat = self.b_to_c_velocity(self.nu_B_hat)

        self.last_imu_innovation = (imu_innovation.copy())
        
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

        if not self.validate_camera_measurement(z_camera):
            self.get_logger().warn("Invalid stereo feature measurement.")
            return

        if not self.ukf_initialized:
            self.initialize_ukf_from_camera(z_camera)
            return

        sigma_camera = self.generate_sigma_points(self.ukf_x, self.ukf_P)

        self.ukf_x, self.ukf_P, cam_innovation, S_cam, K_cam = self.ukf_update_camera(self.ukf_x, self.ukf_P, sigma_camera, z_camera)

        self.s_hat = self.ukf_x[self.idx_s].copy()
        self.nu_B_hat = self.ukf_x[self.idx_nu].copy()
        self.b_a_hat = self.ukf_x[self.idx_ba].copy()
        self.b_g_hat = self.ukf_x[self.idx_bg].copy()

        self.nu_C_hat = self.b_to_c_velocity(self.nu_B_hat)

        self.last_camera_innovation = (cam_innovation.copy())

        acc_camera_B = self.acc_camera_B
        acc_fcu_B = self.acc_fcu_B

        z = self.build_ukf_measurement(measurement, acc_camera_B, acc_fcu_B)

        tau = self.compute_control_tau(distance, e_norm, e_pixel_img, dt)
        self.tau_ukf = tau.reshape(6,1)
        self.publish_torque(self.torque_pub, "body", msg.header.stamp, tau)
        self.publish_torque(self.tau_P_pub, "body", msg.header.stamp, self.tau_P)
        # self.publish_torque(self.tau_D_pub, "body", msg.header.stamp, self.tau_D)
        # self.publish_torque(self.tau_L_pub, "body", msg.header.stamp, self.tau_L)
        # self.publish_torque(self.tau_gamma_pub, "body", msg.header.stamp, self.tau_gamma)
        pwm = self.compute_force_pwm(tau)

        # self.get_logger().info(f"Output tau_P:\n{self.tau_P}", throttle_duration_sec=1.0)
        # self.get_logger().info(f"Output tau_D:\n{self.tau_D}", throttle_duration_sec=1.0)
        # self.get_logger().info(f"Output tau_L:\n{self.tau_L}", throttle_duration_sec=1.0)
        # self.get_logger().info(f"Output tau_\gamma:\n{self.tau_gamma}", throttle_duration_sec=1.0)
        # self.get_logger().info(f"Output tau:\n{self.tau_ukf}", throttle_duration_sec=1.0)

        # self.publish_rc()
        # self.log_debug(tau, pwm)

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
    def interaction_matrix_feature_2d(self, x, y, delta):

        return np.array([
            [-delta/bline,       0,          delta * x/bline,           x*y,      -(1 + x*x),  y],
            [0,            -delta/bline,     delta * y/bline,         1 + y*y,       -x*y,    -x]
        ])

    # =========================================================
    def interaction_matrix_feature_3d(self, x, y, delta):

        return np.array([
            [-delta/bline,       0,          delta * x/bline,           x*y,      -(1 + x*x),  y],
            [0,            -delta/bline,     delta * y/bline,         1 + y*y,       -x*y,    -x],
            [0,                  0,       (delta * delta) / bline,    delta * y,  -delta * x,  0]
        ])

    # =========================================================
    def build_interaction_matrix(self, state):
        state = np.asarray(state).flatten()
        rows = []

        if self.use_3d_matrix_feature:
            for i in range(self.N):
                idx = 3 * i
                x = state[idx]
                y = state[idx + 1]
                Z = state[idx + 2]

                Li = self.interaction_matrix_3d(x, y, Z)
                rows.append(Li)

        else:
            for i in range(self.N):
                idx = 2 * i
                x = state[idx]
                y = state[idx + 1]
                Z = self.last_delta[i]

                Li = self.interaction_matrix_2d(x, y, Z)
                rows.append(Li)

        return np.vstack(rows)

    # =========================================================
    def compute_image_error(self, msg):
        e_pixel = []
        e_pixel_img = []
        e_norm = []
        measurement = []
        deltas = []


        pts = np.array([[p.x, p.y, p.z] for p in msg.polygon.points], dtype=np.float64)
        for i in range(self.N):
            u, v = self.detected_uv[i]
            Z = pts[i, 2]
            if not np.isfinite(Z) or Z <= 1e-4:
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
                measurement.extend([x, y])
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
    def build_Q(self, dt):
        Q = np.zeros((self.nx, self.nx), dtype=np.float64)

        Q[self.idx_s, self.idx_s] = (np.eye(self.n_ft) * self.q_feature_adaptive)
        Q[self.idx_nu,self.idx_nu] = (np.eye(6) * self.q_velocity_adaptive)
        Q[self.idx_ba,self.idx_ba] = (np.eye(3) * self.sigma_ba_rw**2 * dt)
        Q[self.idx_bg,self.idx_bg] = (np.eye(3) * self.sigma_bg_rw**2 * dt)

        return Q

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
    def ukf_process_model(self, state, dt, tau):
        state = np.asarray(state, dtype=np.float64).reshape(self.nx)

        s, nu_B, b_a, b_g = self.unpack_state(state)

        if self.use_3d_matrix_feature:
            L_sigma = self.build_interaction_matrix(s)
        
        else:
            L_sigma = self.build_interaction_matrix(s, self.last_delta)

        nu_C = self.b_to_c_velocity(nu_B)

        s_next = s + dt * (L_sigma @ nu_C)
        nu_next = nu_B + dt * self.fossen_acceleration(nu_B, tau)

        b_a_next = b_a
        b_g_next = b_g

        return self.pack_state(s_next, nu_next, b_a_next, b_g_next)

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

        return state[self.idx_s].copy()

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
    def imu_measurement_model(self, state, tau, R_BN):
        state = np.asarray(state, dtype=np.float64).reshape(self.nx)

        s, nu_B, b_a, b_g = self.unpack_state (state)
        v_B = nu_B[:3]
        omega_B = nu_B[3:]

        nu_dot_B = self.fossen_acceleration(nu_B, tau)
        v_dot_B = nu_dot_B[:3]

        g_N = np.array([0.0, 0.0, 9.80665],dtype=np.float64)
        g_B = R_BN @ g_N

        a_pred_B = (v_dot_B + np.cross(omega_B, v_B) - g_B + b_a)
        omega_pred_B = (omega_B + b_g)

        z_pred = np.concatenate([a_pred_B, omega_pred_B])

        return z_pred

    # =========================================================
    def predict_imu_sigma_points(self, sigma_points, tau, R_BN):
        z_sigma = np.zeros((self.n_sigma, 6), dtype=np.float64)

        for i in range(self.n_sigma):
            z_sigma[i] = (self.imu_measurement_model( sigma_points[i], tau, R_BN))

        return z_sigma
    
    # =========================================================
    def ukf_update_imu(self, x_pred, P_pred, sigma_pred, z_imu, tau, R_BN):
        z_imu = np.asarray(z_imu, dtype=np.float64).reshape(6)
        z_sigma = self.predict_imu_sigma_points(sigma_pred, tau, R_BN)
        z_mean = self.weighted_mean(z_sigma)

        S = self.measurement_covariance(z_sigma, z_mean, self.R_imu)
        Pxz = self.cross_covariance(sigma_pred, x_pred, z_sigma, z_mean)
        K = np.linalg.solve(S, Pxz.T).T

        innovation = z_imu - z_mean
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
    def initialize_ukf_from_camera(self, z_camera):
        z_camera = np.asarray(z_camera, dtype=np.float64).reshape(self.n_ft)

        s0 = z_camera.copy()
        nu_B0 = np.zeros(6)
        b_a0 = np.zeros(3)
        b_g0 = np.zeros(3)
        self.ukf_x = self.pack_state(s0, nu_B0, b_a0, b_g0)

        P0 = np.zeros((self.nx, self.nx), dtype=np.float64)
        P0[self.idx_s, self.idx_s] = (np.eye(self.n_ft) * 1e-4)
        P0[self.idx_nu, self.idx_nu] = (np.eye(6) * 1e-2)
        P0[self.idx_ba, self.idx_ba] = (np.eye(3) * 1e-2)
        P0[self.idx_bg, self.idx_bg] = (np.eye(3) * 1e-4)

        self.ukf_P = P0
        self.ukf_initialized = True
        self.get_logger().info("UKF initialized from stereo camera measurement.")





    # =========================================================
    def ukf_update(self, z):
        z = np.asarray(z).flatten()
        nz = len(z)

        Z_sigma = np.zeros((2*self.nx+1,nz))

        for i in range(2*self.nx+1):
            Z_sigma[i] = self.ukf_measurement_model(
                self.ukf_sigma_pred[i])

        z_pred = np.zeros(nz)

        for i in range(2*self.nx+1):
            z_pred += self.Wm[i]*Z_sigma[i]

        S = np.zeros((nz,nz))
        Pxz = np.zeros((self.nx,nz))

        for i in range(2*self.nx+1):
            dz = Z_sigma[i]-z_pred
            dx = self.ukf_sigma_pred[i]-self.ukf_x.flatten()
            S += self.Wc[i]*np.outer(dz,dz)
            Pxz += self.Wc[i]*np.outer(dx,dz)

        S += self.ukf_R
        K = Pxz @ np.linalg.inv(S)
        innovation = z-z_pred

        self.ukf_x += (K @ innovation.reshape(-1,1))
        self.ukf_P -= K @ S @ K.T
        self.ukf_P = 0.5*(self.ukf_P + self.ukf_P.T)
        self.ukf_P += 1e-9*np.eye(self.nx)

        #Adaptive Q Matrix
        dx = (self.ukf_x - self.x_pred).flatten()

        df = dx[:self.feature_dim]
        dv = dx[self.feature_dim:]

        Qf = np.outer(df, df)
        Qv = np.outer(dv, dv)

        self.ukf_Q[:self.feature_dim,:self.feature_dim] = ((1-self.betaQ_feature) * self.ukf_Q[:self.feature_dim,:self.feature_dim] + self.betaQ_feature * Qf)
        self.ukf_Q[-6:,-6:] = ((1-self.betaQ_velocity) * self.ukf_Q[-6:,-6:] + self.betaQ_velocity * Qv)

        # Adaptive R Matrix
        dz = innovation.reshape(-1,1)

        self.ukf_R = ((1-self.betaR) * self.ukf_R + self.betaR * (dz @ dz.T))

        self.ukf_Q = 0.5*(self.ukf_Q + self.ukf_Q.T)
        self.ukf_R = 0.5*(self.ukf_R + self.ukf_R.T)

        diagQ = np.clip(np.diag(self.ukf_Q),1e-8,1e-2)
        self.ukf_Q = np.diag(diagQ)

        diagR = np.clip(np.diag(self.ukf_R),1e-7,1e-2)
        self.ukf_R = np.diag(diagR)

        # Logging Variables
        pred_error = np.linalg.norm(innovation)
        Pnu = self.ukf_P[-6:, -6:]
        gain_norm = np.linalg.norm(K)
        
        self.nu_B_hat = self.ukf_x[-6:].copy()

        return innovation, Pnu, gain_norm, pred_error, z

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
        S = self.skew(P_BC)
        T_bc = np.block([
            [R_CB,           -S @ R_CB],
            [np.zeros((3,3)),     R_CB]])

        return T_bc

    # =========================================================
    def b_to_c_velocity(self, nu_B):
        S = self.skew(self.P_BC)

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
    def camera_acc_to_body(self, acc_camera):

        return R_BC @ acc_camera

    # =========================================================
    def compute_damping(self, nu):
        nu = nu.flatten()
        linear = self.Dlin @ nu
        quadratic = self.Dquad @ (np.abs(nu) * nu)

        return (linear + quadratic).reshape(-1,1)

    def compute_coriolis(self, nu):
        u,v,w,p,q,r = nu.flatten()
        C = np.array([
            [0,0,0,0,self.m*w,-self.m*v],
            [0,0,0,-self.m*w,0,self.m*u],
            [0,0,0,self.m*v,-self.m*u,0],

            [0,self.m*w,-self.m*v,
             0,self.Izz*r,-self.Iyy*q],
            [-self.m*w,0,self.m*u,
                -self.Izz*r,0,self.Ixx*p],
            [self.m*v,-self.m*u,0,
                self.Iyy*q,-self.Ixx*p,0]])

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
        alpha = self.compute_alpha(self.L_hat)
        gamma = self.compute_gamma(self.nu_B_hat.reshape(-1,1))

        e_dot_hat = self.L_hat @ self.T_bc @ self.nu_B_hat.reshape(-1,1)
        l_dot = self.L_dot @ self.T_bc @ self.nu_B_hat.reshape(-1,1)

        alpha_inv = np.linalg.pinv(alpha)

        tau_P = - self.Kp @ alpha_inv @ e_norm
        tau_D = - self.Kd @ alpha_inv @ e_dot_hat
        tau_L = - alpha_inv @ l_dot
        tau_gamma = alpha_inv @ alpha @ gamma

        # tau = tau_P + tau_D + tau_L + tau_gamma
        tau = tau_P

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
        alpha = self.compute_alpha(L_hat)
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

        Wb = R_BC @ w
        Vb = R_BC @ v + np.cross(Wb, P_BC.reshape(3))

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
    def ukf_logging(self, innovation, Pnu, gain_norm, pred_error, z):
        self.get_logger().info(f"innovation = {np.linalg.norm(innovation):.4f}",throttle_duration_sec=1.0)
        self.get_logger().info(f"P(nu) = {np.sqrt(np.diag(Pnu))}",throttle_duration_sec=1.0)
        self.get_logger().info(f"||K|| = {gain_norm:.3e}",throttle_duration_sec=1.0)
        self.get_logger().info(f"prediction error = {pred_error:.6f}",throttle_duration_sec=1.0)
        self.get_logger().info(f"measurement = {np.linalg.norm(z):.5f}",throttle_duration_sec=1.0)

        sigma = np.sqrt(np.diag(Pnu))

        msg = Float32MultiArray()
        msg.data = [
            float(np.linalg.norm(innovation)),
            float(gain_norm),
            float(pred_error),
            float(np.linalg.norm(z)),
            *sigma.tolist()
        ]

        self.ukf_data_pub.publish(msg)

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



