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

        self.nu_hat_pub = self.create_publisher(TwistStamped, "/ibvs/nu_hat", 10)
        self.torque_pub = self.create_publisher(WrenchStamped, "/ibvs/torque", 10)

        self.err_px_pub = self.create_publisher(Float32MultiArray, "/ibvs/error_px", 10)
        self.err_no_pub = self.create_publisher(Float32MultiArray, "/ibvs/error_no", 10)

        # ---------------- State ----------------
        self.use_3d_matrix_feature = False
        # self.compute_control = self.compute_control_ibvs1

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

        self.current_pwm = [1500] * 18
        self.depth_img = None
        self.detected_uv = None
        self.last_tag_time = None
        self.prev_L = None
        
        # PI controller state
        self.last_time = None
        self.e_integral = None

        ##Tuneable Variables
        self.HEAVE_BIAS = 0 

                     # Sway - Heave - Surge - Pitch - Yaw - Roll
        self.Kp = np.diag([0.7,0.2,0.6,0.3,0.2,0.3]) 
        self.Ki = np.diag([0.02,0.0,0.0,0.01,0.01,0.01])
        
        self.tag_lost = True
        self.TAG_TIMEOUT = 1  # seconds

        self.TcB = self.camera_body_adjoint()

        # # RC command buffer (IMPORTANT)
        # self.rc_cmd = [1500] * 18

        # ---------------- Timers ----------------
        self.create_timer(0.1, self.tag_watchdog)
        self.create_timer(1.0/25.0, self.publish_rc)
        self.get_logger().info(f"IBVS Control {'3D Matrix' if self.use_3d_matrix_feature else '2D Matrix'}")

        self.desired_pts, R = self.compute_desired_corners_pixel(Z_DES=Z_DES, pitch_deg=PITCH_DES_DEG, yaw_deg=YAW_DES_DEG, roll_deg=ROLL_DES_DEG)

        self.desired_normal = R @ np.array([0.0, 0.0, 1.0])

        self.desired_pitch = np.arctan2(
            -self.desired_normal[1],
            self.desired_normal[2]
        )

        self.desired_yaw = np.arctan2(
            self.desired_normal[0],
            self.desired_normal[2]
        )

        p0 = self.desired_pts[3]   # bottom-left
        p1 = self.desired_pts[2]   # bottom-right

        self.desired_roll = np.arctan2(
            p1[1] - p0[1],
            p1[0] - p0[0]
        )

        self.N = 4

        self.n_ft = 3 * self.N if self.use_3d_matrix_feature else 2 * self.N
        self.nx = self.n_ft + 6

        # ---------- UKF Parameters ----------
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0

        self.lambda_ = self.alpha**2 * (self.nx + self.kappa) - self.nx
        self.gamma = np.sqrt(self.nx + self.lambda_)

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
        Q[self.feature_dim:, self.feature_dim:] = np.eye(6) * 5e-4 # velocity process noise

        self.ukf_x = np.zeros((self.nx,1))
        self.ukf_P = np.eye(self.nx)*0.01
        self.ukf_Q = Q
        self.ukf_R = np.eye(self.n_ft)*1e-4
        self.nu_hat = np.zeros((6,1))

    @property
    def feature_dim(self):
        return 3 * self.N if self.use_3d_matrix_feature else 2 * self.N

    # =========================================================
    def cb_corners(self, msg):
        if self.detected_uv is None:
            return
    
        if len(msg.polygon.points) != 4:
            return

        self.last_tag_time = self.get_clock().now()
        self.tag_lost = False

        result = self.compute_image_error(msg)
        if result is None:
            return

        L, distance, e_pixel, e_norm, measurement, e_pixel_img, deltas = result

        self.get_logger().info(f"detection_uv:\n{self.detected_uv}", throttle_duration_sec=1.0)
        self.get_logger().info(f"desired_uv:\n{self.desired_pts}", throttle_duration_sec=1.0)

        dt = self.get_dt()

        self.ukf_predict(dt, deltas)
        innovation, Pnu, gain_norm, pred_error, z  = self.ukf_update(measurement)
        
        self.nu_hat
        self.get_logger().info(
            "UKF velocity = [{}]".format(
                ", ".join(f"{v:.3f}" for v in self.nu_hat.flatten())
            ),
            throttle_duration_sec=1.0
        )

        # self.get_logger().info(f"innovation = {np.linalg.norm(innovation):.4f}",throttle_duration_sec=1.0)
        # # self.get_logger().info(f"P(nu) = {np.sqrt(np.diag(Pnu))}",throttle_duration_sec=1.0)
        # # self.get_logger().info(f"||K|| = {gain_norm:.3e}",throttle_duration_sec=1.0)
        # # self.get_logger().info(f"prediction error = {pred_error:.6f}",throttle_duration_sec=1.0)
        # # self.get_logger().info(f"measurement = {np.linalg.norm(z):.5f}",throttle_duration_sec=1.0)

        Ldot = self.compute_Ldot(L, dt)

        tau = self.compute_control_tau1(L, Ldot, distance, e_norm, e_pixel_img, dt)
        self.publish_torque(self.torque_pub, "body", msg.header.stamp, tau)
        pwm = self.compute_force_pwm(tau)


        # Vc = self.compute_control_ibvs1(L, distance, e_norm, measurement, e_pixel_img, dt)
        # self.publish_twist(self.vel_cam_pub, "camera", msg.header.stamp, Vc[:3], Vc[3:])
        # self.publish_twist(self.vel_body_pub, "body", msg.header.stamp, Vb, Wb)
        # Vb, Wb = self.camera_to_body(Vc)
        # pwm = self.compute_vel_pwm(Vb, Wb)

        self.publish_rc()
        self.publish_error(e_pixel, e_norm)

        self.log_debug(tau, pwm)

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
    def ukf_sigma_points(self):
        A = np.linalg.cholesky(self.ukf_P)
        sigma = np.zeros((2*self.nx+1, self.nx))
        sigma[0] = self.ukf_x.flatten()

        for i in range(self.nx):
            sigma[i+1] = (self.ukf_x.flatten() + self.gamma*A[:,i])
            sigma[self.nx+i+1] = (self.ukf_x.flatten() - self.gamma*A[:,i])

        return sigma

    # =========================================================
    def cb_detection(self, msg):
        if len(msg.detections) == 0:
            self.detected_uv = None
            return

        det = msg.detections[0]
        self.detected_uv = np.array([[c.x, c.y] for c in det.corners],dtype=np.float64)
        
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
    def build_interaction_matrix(self, state, deltas):
        state = np.asarray(state).flatten()
        rows = []

        if self.use_3d_matrix_feature:
            for i in range(self.N):
                idx = 3 * i
                x = state[idx]
                y = state[idx + 1]
                delta = state[idx + 2]

                Li = self.interaction_matrix_feature_3d(x, y, delta)
                rows.append(Li)

        else:
            for i in range(self.N):
                idx = 2 * i
                x = state[idx]
                y = state[idx + 1]
                delta = deltas[i]

                Li = self.interaction_matrix_feature_2d(x, y, delta)
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
            delta = bline / Z

            ud, vd = self.desired_pts[i]
            xd = (ud - CX)/FX
            yd = (vd - CY)/FY
            delta_des = bline / Z_DES

            deltas.append(delta)
            e_pixel_img.extend([u - ud, v - vd])

            if self.use_3d_matrix_feature: # Matrix 3x6 
                measurement.extend([x, y, delta])
                e_pixel.extend([u - ud, v - vd, delta - delta_des])
                e_norm.extend([x - xd, y - yd, delta - delta_des])
            
            else: # Matrix 2x6
                measurement.extend([x, y])
                e_pixel.extend([u - ud, v - vd])
                e_norm.extend([x - xd, y - yd])

        measurement = np.asarray(measurement,dtype=np.float64)

        L = self.build_interaction_matrix(measurement, deltas)
        distance = np.mean(pts[:,2])

        e_pixel = np.asarray(e_pixel).reshape(-1, 1)
        e_norm = np.asarray(e_norm).reshape(-1, 1)
        measurement = measurement.reshape(-1, 1)
        e_pixel_img = np.asarray(e_pixel_img).reshape(-1, 1)
        deltas = np.asarray(deltas, dtype=np.float64)

        return L, distance, e_pixel, e_norm, measurement, e_pixel_img, deltas

    # =========================================================
    def ukf_process_model(self, state, dt, deltas):
        state = np.asarray(state).flatten()
        feature = state[:self.feature_dim]
        nu = state[self.feature_dim:]

        L = self.build_interaction_matrix(feature, deltas)
        feature_next = feature + dt*(L @ nu)

        x_next = np.concatenate([feature_next, nu])
        return x_next

    # =========================================================
    def ukf_measurement_model(self, state):
        return state[:self.feature_dim]

        # =========================================================
    def ukf_predict(self, dt, deltas):
        sigma = self.ukf_sigma_points()
        sigma_pred = np.zeros_like(sigma)

        for i in range(2*self.nx+1):
            sigma_pred[i] = self.ukf_process_model(
                sigma[i], dt, deltas)

        x_pred = np.zeros(self.nx)

        for i in range(2*self.nx+1):
            x_pred += self.Wm[i] * sigma_pred[i]

        P_pred = np.zeros((self.nx,self.nx))

        for i in range(2*self.nx+1):
            dx = sigma_pred[i]-x_pred
            P_pred += self.Wc[i]*np.outer(dx,dx)

        P_pred += self.ukf_Q
        self.ukf_sigma_pred = sigma_pred
        self.ukf_x = x_pred.reshape(-1,1)
        self.ukf_P = P_pred

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

        pred_error = np.linalg.norm(innovation)

        self.ukf_x += (K @ innovation.reshape(-1,1))
        self.ukf_P -= K @ S @ K.T
        
        Pnu = self.ukf_P[-6:, -6:]
        gain_norm = np.linalg.norm(K)
        
        self.nu_hat = self.ukf_x[-6:].copy()

        return innovation, Pnu, gain_norm, pred_error, z

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
    def compute_Ldot(self, L, dt):
        if self.prev_L is None:
            self.prev_L = L.copy()

            return np.zeros_like(L)

        Ldot = (L - self.prev_L) / dt
        self.prev_L = L.copy()

        return Ldot

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
        TcB = np.block([
            [R_BC,         S @ R_BC],
            [np.zeros((3,3)),  R_BC]])

        return TcB

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
    def compute_gamma(self):
        nu = self.nu_hat
        gamma = (self.compute_coriolis(nu) +
                self.compute_damping(nu) +
                self.compute_restoring())

        return gamma

    # =========================================================
    def compute_alpha(self, L):

        return L @ self.TcB @ self.Minv

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
    def compute_control_tau1(self, L, Ldot, distance, e_norm, e_pixel, dt):
        A = L.T @ L + mu**2 * np.eye(6)

        if self.e_integral is None:
            self.e_integral = np.zeros_like(e_norm)

        self.e_integral += e_norm * dt
        self.e_integral = np.clip(self.e_integral, -0.3, 0.3)

        alpha = self.compute_alpha(L)
        gamma = self.compute_gamma()

        rhs = lambda_gain * e_norm + Ldot @ self.nu_hat - alpha @ gamma
        tau = -np.linalg.pinv(alpha) @ rhs

        if np.max(np.abs(e_pixel)) < dead_band:
            tau[:] = 0

        self.limit_force(tau)
        tau = tau.flatten()

        return tau

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
            tau[:3] *= MAX_MOMEN / force

        momen = np.linalg.norm(tau[3:])
        if momen > MAX_FORCE:
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
