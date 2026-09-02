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

from .parameter import *

from .control import IBVS_Controller
from .so_control import Stereo_IBVS_Control
from .so_estimator import Stereo_UKF_Estimator
from .estimator import UKF_Estimator

class IBVSRCController(Node):
    def __init__(self):
        super().__init__("IBVS_RC_Controller")

        self.bridge = CvBridge()
        imu_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=25)
        
        # ---------------- Subscribers ----------------
        self.sub_corners = self.create_subscription(PolygonStamped, "/apriltag/corners", self.cb_corners, qos_profile_sensor_data)
        self.sub_detection = self.create_subscription(AprilTagDetectionArray, "/detection1", self.cb_detection_left, 10)
        # self.camera_gyro_sub = self.create_subscription(Imu, '/camera/camera/gyro/sample', self.cb_camera_gyro, 200)
        # self.camera_accel_sub = self.create_subscription(Imu, '/camera/camera/accel/sample', self.cb_camera_accel, 100)
        self.fcu_imu_sub = self.create_subscription(Imu, '/mavros/imu/data_raw', self.cb_fcu_imu, imu_qos)

        # ---------------- Publishers ----------------
        self.rc_override_pub = self.create_publisher(OverrideRCIn, "/mavros/rc/override", 10)
        self.pwm_pub = self.create_publisher(Int16MultiArray, "/ibvs/pwm_debug", 10)

        self.vel_body_pub = self.create_publisher(TwistStamped, "/ibvs/vel_body", 10)
        self.nu_B_hat_pub = self.create_publisher(TwistStamped, "/ibvs/nu_B_hat", 10)
        self.torque_pub = self.create_publisher(WrenchStamped, "/ibvs/torque", 10)

        self.ukf_data_pub = self.create_publisher(Float32MultiArray, "/ibvs/ukf/data", 10)
        self.err_px_pub = self.create_publisher(Float32MultiArray, "/ibvs/error/px", 10)
        self.err_no_pub = self.create_publisher(Float32MultiArray, "/ibvs/error/no", 10)

        self.declare_state()
        self.current_pwm = [1500] * 18


        # ---------- Desired Tag Configuration ----------
        self.desired_pts, R = self.desired_corners(Z_DES=Z_DES, pitch_deg=PITCH_DES_DEG, yaw_deg=YAW_DES_DEG, roll_deg=ROLL_DES_DEG)
        self.desired_normal = R @ np.array([0.0, 0.0, 1.0])

        p0, p1 = self.desired_pts[3], self.desired_pts[2]
        self.desired_roll = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
        self.desired_pitch = np.arctan2(-self.desired_normal[1], self.desired_normal[2])
        self.desired_yaw = np.arctan2(self.desired_normal[0], self.desired_normal[2])


        # ---------- UKF Parameter ----------
        self.N = 4 
        self.n_dim = 3 * self.N if self.use_3d_matrix_feature else 2 * self.N
        self.n_cam = self.n_dim

        self.ukf_state = self.n_cam + 18

        self.tau_ukf = np.zeros((6,1))
        self.nu_B_hat = np.zeros((6,1))
        self.nu_C_hat = np.zeros((6,1))
        self.b_a_hat = np.zeros((3,1))
        self.b_g_hat = np.zeros((3,1))

        self.shared = Shared_State()
        self.geometry = IBVS_Geometry(N=self.N, use_3d_matrix_feature=self.use_3d_matrix_feature,)           

        self.estimator = UKF_Estimator(
            shared=self.shared,
            nx=self.ukf_state,
            feature_dim=self.n_dim,
            N=self.N,
            use_3d_matrix_feature=self.use_3d_matrix_feature,
            use_delta_matrix=self.use_delta_matrix,
            stereo_cam=self.stereo_cam,
            logger=self.get_logger(),)

        self.controller = IBVS_Controller(
            shared=self.shared, 
            N=self.N,
            use_3d_matrix_feature=self.use_3d_matrix_feature,
            use_dls=self.dls_matrix, 
            use_delta_matrix=self.use_delta_matrix,)

    def stamp_to_sec(self, stamp):
        return (float(stamp.sec) + float(stamp.nanosec) * 1e-9)

    def declare_state(self):
        # ---------------------- System Flags ----------------------
        self.use_3d_matrix_feature = True
        self.use_delta_matrix = False
        self.dls_matrix = True
        self.stereo_cam = False

        self.use_camera_ukf = True
        self.tag_was_initialized = False
        self.reset_on_reacquire = False

        # --------------- Perception & Tracking State ---------------
        self.depth_img = None
        self.detected_uv_left = None
        
        # Timestamps
        self.last_tag_time = None
        self.last_imu_time = None
        self.last_camera_time = None
        self.last_control_time = None
        self.last_estimator_dt = None
        self.control_dt = None

        # --------------------- Camera IMU Data ---------------------
        self.acc_camera = None
        self.acc_camera_B = None
        self.gyro_camera = None
        self.acc_camera_stamp = None
        self.gyro_camera_stamp = None

        # ----------------------- FCU IMU Data -----------------------
        self.acc_fcu = None
        self.acc_fcu_B = None
        self.gyro_fcu = None
        self.acc_fcu_stamp = None
        self.gyro_fcu_stamp = None

        # -------------------------- Timers --------------------------
        self.camera_imu_timeshift = 0.00702
        self.TAG_TIMEOUT = 1  # seconds
        self.create_timer(0.1, self.tag_watchdog)
        self.create_timer(1.0/25.0, self.publish_rc)
        # self.create_timer(1.0 / 100.0, self.cb_control)

        # ------------------------- Variables ------------------------
        self.latest_distance_mean = None
        self.e_norm_left = None
        self.e_pixel_left = None
        self.last_imu_innovation = None
        self.last_camera_innovation = None

        self.get_logger().info(f"IBVS Control {'3D Matrix' if self.use_3d_matrix_feature else '2D Matrix'} with feature {'Depth' if not self.use_delta_matrix else 'Delta'}")

    # =========================================================
    def reset_state(self):
        self.controller.reset()
        self.estimator.reset()

        self.shared.last_distance = None
        self.shared.camera_measurement_valid = False
        self.last_camera_innovation = None

        self.get_logger().info("IBVS State variables successfully reset.")

    # =========================================================
    def update_estimator(self):
        self.vB_hat = self.estimator.ukf_x[self.estimator.idx_vB].copy()
        self.wB_hat = self.estimator.ukf_x[self.estimator.idx_wB].copy()
        self.bo_hat = self.estimator.ukf_x[self.estimator.idx_bo].copy()
        self.bg_hat = self.estimator.ukf_x[self.estimator.idx_bg].copy()
        self.aB_hat = self.estimator.ukf_x[self.estimator.idx_aB].copy()
        self.ba_hat = self.estimator.ukf_x[self.estimator.idx_ba].copy()
        
        self.nu_B_hat = np.concatenate([self.vB_hat, self.wB_hat]).reshape(6, 1)

        self.nu_C_hat = self.geometry.T_bc_0 @ self.nu_B_hat
        self.s_hat = self.estimator.ukf_x[self.estimator.idx_s].copy()

    # =========================================================
    def cb_control(self):
        if not self.shared.ukf_initialized or not self.shared.camera_measurement_valid or self.shared.last_distance is None:
            return

        if self.e_norm_left is None or self.e_pixel_left is None:
            return

        now = self.get_clock().now()

        if self.last_control_time is None:
            self.last_control_time = now
            return

        control_dt = (now - self.last_control_time).nanoseconds * 1e-9
        self.last_control_time = now

        if control_dt <= 0.0 or control_dt > 0.2:
            self.get_logger().warn(f"Invalid control dt: {control_dt:.6f} s")
            return

        self.control_dt = control_dt

        distance_mean = self.latest_distance_mean
        tau = self.controller.compute_control_tau_classic( 
                feature_hat=self.s_hat,
                last_distance = self.shared.last_distance,

                nu_B_hat=self.nu_B_hat,
                distance=distance_mean,

                e_norm=self.e_norm_left,
                e_pixel=self.e_pixel_left,

                dt=control_dt,
                tag_lost=self.shared.tag_lost,)

        self.tau_ukf = np.asarray(tau, dtype=float).reshape(6,1)
        self.publish_torque(self.torque_pub, "body", self.get_clock().now().to_msg(), tau)

        pwm = self.controller.compute_force_pwm(tau)
        self.current_pwm = pwm
        
        self.log_debug(tau, self.nu_B_hat, pwm)

    # =========================================================
    def cb_fcu_imu(self, msg):
        q = msg.orientation
        t = self.stamp_to_sec(msg.header.stamp)

        if self.last_imu_time is None:
            self.last_imu_time = t
            return

        dt = t - self.last_imu_time
        self.last_imu_time = t

        if dt <= 0.0 or dt > 0.1:
            self.get_logger().warn(f"Invalid IMU dt: {dt:.6f} s")
            return

        R_NB = self.quaternion_to_rotation(msg.orientation)

        accel_flu = np.array([msg.linear_acceleration.x, 
                              msg.linear_acceleration.y, 
                              msg.linear_acceleration.z])

        gyro_flu = np.array([msg.angular_velocity.x, 
                             msg.angular_velocity.y, 
                             msg.angular_velocity.z])

        accel_B, gyro_B = self.imu_R_to_NED(R_IB, accel_flu, gyro_flu)

        z_imu = np.concatenate([accel_B, gyro_B])

        if not self.shared.ukf_initialized:
            return

        last_distance = self.shared.last_distance
        tau = self.tau_ukf

        x_pred, P_pred, sigma_pred = self.estimator.ukf_predict(self.estimator.ukf_x, self.estimator.ukf_P, dt, last_distance,)
        self.estimator.ukf_x, self.estimator.ukf_P, imu_innovation, S_imu, K_imu, z_imu_mean = self.estimator.ukf_update_imu_fcu(x_pred, P_pred, sigma_pred, z_imu, R_NB)

        self.last_imu_innovation = imu_innovation.copy()
        self.update_estimator()
        self.ukf_logging(source="imu", innovation=imu_innovation, K=K_imu, z=z_imu)

        self.publish_twist(self.nu_B_hat_pub, "nu_B_hat", msg.header.stamp, self.vB_hat, self.wB_hat)
        self.cb_control()
         
    # =========================================================
    def cb_corners(self, msg):
        if self.detected_uv_left is None:
            return
    
        if len(msg.polygon.points) != self.N:
            return
        
        camera_time = self.stamp_to_sec(msg.header.stamp)
        camera_time_ukf = camera_time - self.camera_imu_timeshift

        if self.last_camera_time is None:
            camera_dt = 0.033
        else:
            camera_dt = camera_time_ukf - self.last_camera_time

        self.last_camera_time = camera_time_ukf
        self.last_camera_dt = camera_dt

        tag_reacquired = self.tag_was_initialized and self.shared.tag_lost
 
        self.last_tag_time = self.get_clock().now()
        self.shared.tag_lost = False

        result = self.compute_image_error_stereo(msg)
        if result is None:
            return

        (distance_mean, e_pixel_img, e_pixel_left, e_norm_left, measurement_left) = result

        z_cam = measurement_left.flatten()
        self.publish_error(e_pixel_left, e_norm_left)

        if not self.shared.ukf_initialized:
            self.estimator.initialize_ukf_from_camera(measurement_left)

            self.shared.ukf_initialized = True
            self.tag_was_initialized = True
            self.shared.camera_measurement_valid = True
            self.get_logger().info("UKF initialized from camera.")
            return

        if tag_reacquired:
            self.get_logger().warn("AprilTag REACQUIRED -> resetting UKF/controller state.")
            self.reset_state()

            self.estimator.initialize_ukf_from_camera(measurement_left)

            self.shared.ukf_initialized = True
            self.shared.camera_measurement_valid = True
            self.shared.tag_lost = False

            self.get_logger().info("UKF reinitialized after tag reacquisition.")
            return

        sigma_camera = self.estimator.generate_sigma_points(self.estimator.ukf_x, self.estimator.ukf_P)
        self.estimator.ukf_x, self.estimator.ukf_P, cam_innovation, S_cam, K_cam, z_cam_mean = self.estimator.ukf_update_camera(self.estimator.ukf_x, self.estimator.ukf_P, sigma_camera, z_cam)

        self.last_camera_dt = camera_dt
        self.last_camera_innovation = cam_innovation.copy()

        self.update_estimator()
        self.ukf_logging(source="camera", innovation=cam_innovation, K=K_cam, z=z_cam)

        

        self.latest_distance_mean = distance_mean
        self.e_norm_left = e_norm_left.copy()
        self.e_pixel_left = e_pixel_left.copy()
        self.last_camera_innovation = cam_innovation.copy()
        self.shared.camera_measurement_valid = True

        self.cb_control()





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
    def imu_R_to_NED(self, R, accel, gyro):
        accel  = np.asarray(accel , dtype=np.float64).reshape(3)
        gyro = np.asarray(gyro, dtype=np.float64).reshape(3)

        accel_B = R @ accel 
        gyro_B = R @ gyro

        return accel_B, gyro_B

    # =========================================================
    def cb_detection_left(self, msg):
        if len(msg.detections) == 0:
            self.detected_uv_left = None
            self.shared.last_distance = None
            return

        det = msg.detections[0]
        self.detected_uv_left = np.array([[c.x, c.y] for c in det.corners],dtype=np.float64)

    # =========================================================   
    def pixel_to_norm(self, u, v):
        x = (u - CX)/FX
        y = (v - CY)/FY

        return x, y
    
    # =========================================================
    def compute_image_error_stereo(self, msg):
        e_pixel_left = []
        e_norm_left = []
        measurement_left = []

        depth = []
        deltas = []
        e_pixel_img = []

        pts = np.array([[p.x, p.y, p.z] for p in msg.polygon.points])
        for i in range(4):
            u_l, v_l = self.detected_uv_left[i]
            Z = pts[i, 2]
            if not np.isfinite(Z) or Z <= 0 or Z < 1e-4:
                return None

            x_l, y_l = self.pixel_to_norm(u_l, v_l)
            delta =  bline / Z

            depth.append(Z)
            deltas.append(delta)

            ud_l, vd_l = self.desired_pts[i]

            xd_l, yd_l = self.pixel_to_norm(ud_l, vd_l)
            delta_des = bline / Z_DES

            e_pixel_img.extend([u_l - ud_l, v_l - vd_l])

            if self.use_3d_matrix_feature and self.use_delta_matrix:
                measurement_left.extend([u_l, v_l, delta])
                e_pixel_left.extend([u_l - ud_l, v_l - vd_l, delta - delta_des])
                e_norm_left.extend([x_l - xd_l, y_l - yd_l, delta - delta_des])

            elif self.use_3d_matrix_feature and not self.use_delta_matrix:
                measurement_left.extend([u_l, v_l, Z])
                e_pixel_left.extend([u_l - ud_l, v_l - vd_l, Z - Z_DES])
                e_norm_left.extend([x_l - xd_l, y_l - yd_l, Z - Z_DES])

            elif not self.use_3d_matrix_feature and self.use_delta_matrix:
                measurement_left.extend([u_l, v_l])
                e_pixel_left.extend([u_l - ud_l, v_l - vd_l])
                e_norm_left.extend([x_l - xd_l, y_l - yd_l])

            elif not self.use_3d_matrix_feature and not self.use_delta_matrix:
                measurement_left.extend([u_l, v_l])
                e_pixel_left.extend([u_l - ud_l, v_l - vd_l])
                e_norm_left.extend([x_l - xd_l, y_l - yd_l])

        if not self.use_delta_matrix:
            distance = np.asarray(depth, dtype=np.float64).reshape(4, 1)
            distance_mean = float(np.mean(depth))
        else:
            distance = np.asarray(deltas, dtype=np.float64).reshape(4, 1)
            distance_mean = float(np.mean(deltas))

        e_pixel_left = np.asarray(e_pixel_left).reshape(-1, 1)
        e_norm_left = np.asarray(e_norm_left).reshape(-1, 1)
        measurement_left = np.asarray(measurement_left,dtype=np.float64).reshape(-1, 1)    

        e_pixel_img = np.asarray(e_pixel_img).reshape(-1, 1)        
        self.shared.last_distance = distance.copy()

        return (distance_mean, e_pixel_img,
                e_pixel_left, e_norm_left, measurement_left)

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
        x, y, z, w = q.x, q.y, q.z, q.w

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = np.copysign(np.pi / 2.0, sinp)
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

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

        msg = Float32MultiArray()
        msg.data = [
            float(source_id),
            float(innovation_norm),
            float(gain_norm),
            float(state_correction_norm),
            float(measurement_norm),
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
    def log_debug(self, tau=None, nu_hat=None, pwm=None):
        nu_hat = np.asarray(nu_hat, dtype=np.float64).reshape(-1)
        tau = np.asarray(tau, dtype=np.float64).reshape(-1)
        pwm = np.asarray(pwm, dtype=np.float64).reshape(-1)
        
        if nu_hat is not None:
            nu_hat = np.asarray(nu_hat, dtype=np.float64).reshape(-1)
            self.get_logger().info(
                f"Surge = {nu_hat[0]:.2f} |"
                f"Sway = {nu_hat[1]:.2f} |"
                f"Heave = {nu_hat[2]:.2f} |"
                f"Roll = {nu_hat[3]:.2f} |"
                f"Pitch = {nu_hat[4]:.2f} |"
                f"Yaw = {nu_hat[5]:.2f} |",
                throttle_duration_sec=1.0)
            
        if tau is not None:
            tau = np.asarray(tau, dtype=np.float64).reshape(-1)
            self.get_logger().info(
                f"Surge = {tau[0]:.2f} |"
                f"Sway = {tau[1]:.2f} |"
                f"Heave = {tau[2]:.2f} |"
                f"Roll = {tau[3]:.2f} |"
                f"Pitch = {tau[4]:.2f} |"
                f"Yaw = {tau[5]:.2f} ",
                throttle_duration_sec=1.0)
        
        if pwm is not None:
            pwm = np.asarray(pwm, dtype=np.float64).reshape(-1)
            self.get_logger().info(
                f"Surge={pwm[4]} |"
                f"Sway={pwm[5]} |"
                f"Heave={pwm[2]} |"
                f"Roll={pwm[1]} |"
                f"Pitch={pwm[0]} |"
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
        err_px_msg.data = e_pixel.astype(np.float32).ravel().tolist()
        self.err_px_pub.publish(err_px_msg)

        err_no_msg = Float32MultiArray()
        err_no_msg.data = e_norm.astype(np.float32).ravel().tolist()
        self.err_no_pub.publish(err_no_msg)
        
    # =========================================================
    def tag_watchdog(self):
        if self.last_tag_time is None:
            return

        dt = (self.get_clock().now() - self.last_tag_time).nanoseconds * 1e-9

        if dt > self.TAG_TIMEOUT:
            if not self.shared.tag_lost:
                self.shared.tag_lost = True
                self.shared.camera_measurement_valid = False
                self.get_logger().warn("AprilTag LOST")
            
            # Only reset to neutral IF the tag is actually lost
            self.current_pwm = [1500] * 18
            self.current_pwm[2] = 1500 + 100
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
