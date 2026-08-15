import numpy as np
from .parameter import *

class IBVS_Controller:
    def __init__(self, N=4, use_3d_matrix_feature=True,):
        self.use_3d_matrix_feature = use_3d_matrix_feature
        self.N = N

        self.geometry = IBVS_Geometry(N=self.N, use_3d_matrix_feature=self.use_3d_matrix_feature,)
        self.shared = Shared_State()

        self.T_bc = self.geometry.T_bc
        self.Minv = np.linalg.inv(M)

                # Camera PID Sway - Heave - Surge - Pitch - Yaw - Roll

                  # Tau PID Surge - Sway - Heave - Roll - Pitch - Yaw
        self.Kp = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) 
        self.Kd = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.Ki = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        self.L_prev = None
        self.L_hat = None
        self.L_dot = None
        self.e_integral = None

    # =========================================================
    @property
    def feature_dim(self):
        return 3 * self.N if self.use_3d_matrix_feature else 2 * self.N

    # =========================================================
    def reset(self):
        self.L_prev = None
        self.L_hat = None
        self.L_dot = None
        self.e_integral = None
       
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
    def compute_alpha(self, L):
        return L @ self.T_bc @ self.Minv
    
    # =========================================================
    def compute_gamma(self, nu):
        gamma = (self.compute_coriolis(nu) +
                self.compute_damping(nu) +
                self.compute_restoring())

        return gamma

    # =========================================================
    def update_L_hat(self, feature_hat, last_delta=None, tag_lost=False):
        if tag_lost and self.L_hat is not None:
            return self.L_hat
        
        feature_hat = np.asarray(feature_hat, dtype=float).flatten()

        if self.use_3d_matrix_feature:
            self.L_hat = self.geometry.build_interaction_matrix(feature_hat)

        else:
            if last_delta is None:
                self.L_hat = None
                return

            self.L_hat = self.geometry.build_interaction_matrix(feature_hat, last_delta)

        return self.L_hat

    # =========================================================
    def compute_Ldot(self, dt):
        if self.L_hat is None:
            self.L_dot = None
            return None

        if dt <= 1e-6:
            self.L_dot = np.zeros_like(self.L_hat)
            return self.L_dot

        if self.L_prev is None:
            self.L_prev = self.L_hat.copy()
            self.L_dot = np.zeros_like(self.L_hat)
            return self.L_dot

        self.L_dot = (self.L_hat - self.L_prev) / dt
        self.L_prev = self.L_hat.copy()
        return self.L_dot

    # =========================================================
    def compute_Ldot_analytical(self, feature_hat, nu_B_hat, last_delta=None):
        if self.L_hat is None or nu_B_hat is None:
            self.L_dot = None
            return None

        feature_hat = np.asarray(feature_hat, dtype=float).flatten()
        nu_B_hat = np.asarray(nu_B_hat, dtype=float).reshape(6)
        camera_nu = self.T_bc @ nu_B_hat.reshape(6, 1)
        feature_dot = self.L_hat @ camera_nu

        Ldot_rows = []

        if self.use_3d_matrix_feature:
            for i in range(self.N):
                idx = 3 * i
                x, y, delta = feature_hat[idx:idx + 3]
                x_dot = feature_dot[idx, 0]
                y_dot = feature_dot[idx + 1, 0]
                delta_dot = feature_dot[idx + 2, 0]

                Ldot_rows.append(
                    self.geometry.interaction_matrix_dot(
                        x, y, delta, x_dot, y_dot, delta_dot
                    )
                )
        else:
            if last_delta is None:
                self.L_dot = None
                return None

            last_delta = np.asarray(last_delta, dtype=float).reshape(self.N)
            for i in range(self.N):
                idx = 2 * i
                x, y = feature_hat[idx:idx + 2]
                delta = last_delta[i]
                x_dot = feature_dot[idx, 0]
                y_dot = feature_dot[idx + 1, 0]

                L3 = self.interaction_matrix_3d(x, y, delta)
                delta_dot = float(L3[2, :] @ camera_nu.flatten())

                Ldot_rows.append(
                    self.geometry.interaction_matrix_dot(
                        x, y, delta, x_dot, y_dot, delta_dot
                    )
                )

        self.L_dot = np.vstack(Ldot_rows)
        return self.L_dot
    
    # =========================================================
    def compute_control_tau_debug(self, feature_hat, last_delta, nu_B_hat, distance, e_norm, e_pixel, dt, tag_lost=False,):
        _ = distance
        self.update_L_hat(feature_hat, last_delta, tag_lost)
        self.compute_Ldot(dt)

        if self.L_hat is None or self.L_dot is None:
            return np.zeros(6)

        nu_B_hat = np.asarray(nu_B_hat, dtype=float).reshape(6, 1)
        e_norm = np.asarray(e_norm, dtype=float).reshape(-1, 1)
        e_pixel = np.asarray(e_pixel, dtype=float).reshape(-1, 1)

        alpha = self.compute_alpha(self.L_hat)
        gamma = self.compute_gamma(nu_B_hat)
        e_dot_hat = self.L_hat @ self.T_bc @ nu_B_hat
        l_dot = self.L_dot @ self.T_bc @ nu_B_hat
        alpha_inv = np.linalg.pinv(alpha)

        tau_P = -self.Kp @ alpha_inv @ e_norm
        tau_D = -self.Kd @ alpha_inv @ e_dot_hat
        tau_L = -alpha_inv @ l_dot
        tau_gamma = alpha_inv @ alpha @ gamma

        tau = tau_P + tau_D + tau_L + tau_gamma

        if np.max(np.abs(e_pixel)) < dead_band:
            tau[:] = 0.0

        self.limit_force(tau)

        return tau.reshape(6)

    # =========================================================
    def compute_control_tau_DLS(self, feature_hat, last_delta, nu_B_hat, distance, e_norm, e_pixel, dt, tag_lost=False,):
        _ = distance
        self.update_L_hat(feature_hat, last_delta, tag_lost)
        self.compute_Ldot(dt)

        if self.L_hat is None or self.L_dot is None:
            return np.zeros(6)

        nu_B_hat = np.asarray(nu_B_hat, dtype=float).reshape(6, 1)
        e_norm = np.asarray(e_norm, dtype=float).reshape(-1, 1)
        e_pixel = np.asarray(e_pixel, dtype=float).reshape(-1, 1)

        alpha = self.compute_alpha(self.L_hat)
        gamma = self.compute_gamma(nu_B_hat)

        A = alpha.T @ alpha + mu**2 * np.eye(6)

        e_dot_hat = self.L_hat @ self.T_bc @ nu_B_hat
        l_dot = self.L_dot @ self.T_bc @ nu_B_hat
        gams = alpha @ gamma

        tau_P = - self.Kp @ np.linalg.solve(A, alpha.T @ e_norm).reshape(6)
        tau_D = - self.Kd @ np.linalg.solve(A, alpha.T @ e_dot_hat).reshape(6)
        tau_L = - np.linalg.solve(A, alpha.T @ l_dot).reshape(6)
        tau_gamma = np.linalg.solve(A, alpha.T @ gams).reshape(6)

        tau = tau_P + tau_D + tau_L + tau_gamma

        if np.max(np.abs(e_pixel)) < dead_band:
            tau[:] = 0

        self.limit_force(tau)

        return tau.reshape(6)

    # =========================================================
    def compute_control_ibvs1(self, L, distance, e_norm, e_pixel, dt):
        _ = distance
        L = np.asarray(L, dtype=float)
        e_norm = np.asarray(e_norm, dtype=float).reshape(-1, 1)
        e_pixel = np.asarray(e_pixel, dtype=float).reshape(-1, 1)

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

        return Vc.reshape(6)

    # =========================================================
    def compute_control_ibvs2(self, L, distance, e_norm, pixel_norm, detected_uv, desired_roll):
        L_xy = L[:,[0,1,3,4]]
        L_z  = L[:,[2,5]]

        p1 = np.asarray(detected_uv[0])
        p2 = np.asarray(detected_uv[1])
        measured_roll = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

        depth_error = bline/ distance - bline/ Z_DES
        roll_error = np.arctan2(np.sin(measured_roll - desired_roll), 
                                np.cos(measured_roll - desired_roll))

        vz = -self.kz * depth_error
        wz = -self.kw * roll_error

        A = L_xy.T @ L_xy + mu**2*np.eye(4)
        b_p = L_xy.T @ e_norm
        v_p = np.linalg.solve(A, b_p).flatten()

        rhs = self.lambda_gain * e_norm + L_z @ np.array([[vz],[wz]])
        b = L_xy.T @ rhs
        Vxy = - np.linalg.solve(A,b)

        Vc = np.array([
            Vxy[0, 0], Vxy[1, 0], vz,
            Vxy[2, 0], Vxy[3, 0], wz,
        ]).reshape(6, 1)

        if np.max(np.abs(pixel_norm)) < dead_band:
            Vc[:] = 0

        self.limit_velocity(Vc)
        return Vc.reshape(6)

    # =========================================================
    def compute_control_ibvs3(self, L, distance, e_norm, pixel_norm):
        _ = distance

    # =========================================================
    def compute_control_ibvs4(self, L, distance, e_norm, pixel_norm):
        _ = distance

    # =========================================================
    def limit_velocity(self, Vc):
        Vc = np.asarray(Vc)
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

        moment = np.linalg.norm(tau[3:])
        if moment > MAX_MOMEN:
            tau[3:] *= MAX_MOMEN / moment

    # =========================================================
    def force_to_pwm(self, F, bias=0):
        return int(np.clip(1500 + F * 15 + bias, 1100, 1900))

    # =========================================================
    def compute_force_pwm(self, tau, heave_bias=0):
        tau = np.asarray(tau, dtype=float).reshape(6)
        pwm = [1500] * 18
        pwm[4] = self.force_to_pwm(tau[0])
        pwm[5] = self.force_to_pwm(tau[1])
        pwm[2] = self.force_to_pwm(tau[2], heave_bias)
        pwm[1] = self.force_to_pwm(tau[3])
        pwm[0] = self.force_to_pwm(tau[4])
        pwm[3] = self.force_to_pwm(tau[5])
        return pwm

    # =========================================================
    def compute_vel_pwm(self, Vb, Wb, heave_bias=0):
        Vb = np.asarray(Vb, dtype=float).reshape(3)
        Wb = np.asarray(Wb, dtype=float).reshape(3)

        pwm = [1500] * 18
        pwm[4] = int(np.clip(1500 + 400 * Vb[0], 1100, 1900))
        pwm[5] = int(np.clip(1500 + 400 * Vb[1], 1100, 1900))
        pwm[2] = int(np.clip(1500 + 400 * Vb[2] + heave_bias, 1100, 1900))
        pwm[1] = int(np.clip(1500 + 400 * Wb[0], 1100, 1900))
        pwm[0] = int(np.clip(1500 + 400 * Wb[1], 1100, 1900))
        pwm[3] = int(np.clip(1500 + 400 * Wb[2], 1100, 1900))
        return pwm
