import numpy as np
from .parameter import *

class Stereo_IBVS_Control:
    def __init__(self, shared, N=4, use_3d_matrix_feature=True, use_dls=False, use_delta_matrix=False,):
        self.use_3d_matrix_feature = use_3d_matrix_feature
        self.use_delta_matrix = use_delta_matrix
        self.dls_matrix = use_dls
        self.N = N
        self.shared = shared

        self.geometry = IBVS_Geometry(N=self.N, use_3d_matrix_feature=self.use_3d_matrix_feature,)

        self.Minv = np.linalg.inv(M)

                # Camera PID Sway - Heave - Surge - Pitch - Yaw - Roll

                  # Tau PID Surge - Sway - Heave - Roll - Pitch - Yaw
        self.Kp = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) 
        self.Kd = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.Ki = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])


        self.L_hat_left = None
        self.L_hat_right = None
        self.L_dot_left = None
        self.L_dot_right = None
        self.e_integral = None

    # =========================================================
    @property
    def feature_dim(self):
        return 3 * self.N if self.use_3d_matrix_feature else 2 * self.N

    # =========================================================
    def reset(self):
        self.L_hat_left = None
        self.L_hat_right = None
        self.L_dot_left = None
        self.L_dot_right = None
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
    def compute_alpha_so(self, L):
        return L @ self.Minv
    
    # =========================================================
    def compute_gamma(self, nu):
        gamma = (self.compute_coriolis(nu) +
                self.compute_damping(nu) +
                self.compute_restoring())

        return gamma

    # =========================================================
    def update_L_hat(self, feature_hat_left, feature_hat_right, last_depth=None, last_delta=None, tag_lost=False):
        if tag_lost:
            if self.L_hat_left is not None and self.L_hat_right is not None:
                return self.L_hat_left, self.L_hat_right
            return None, None

        if (not self.use_3d_matrix_feature) and self.use_delta_matrix:
            if last_delta is None:
                self.L_hat_left = None
                self.L_hat_right = None
                return None, None

        if (not self.use_3d_matrix_feature) and (not self.use_delta_matrix):
            if last_depth is None:
                self.L_hat_left = None
                self.L_hat_right = None
                return None, None
        
        feature_hat_left = np.asarray(feature_hat_left, dtype=float).flatten()
        feature_hat_right = np.asarray(feature_hat_right, dtype=float).flatten()

        if self.use_3d_matrix_feature and self.use_delta_matrix:
            self.L_hat_left = self.geometry.build_interaction_matrix_delta(feature_hat_left)
            self.L_hat_right = self.geometry.build_interaction_matrix_delta(feature_hat_right)

        elif self.use_3d_matrix_feature and not self.use_delta_matrix:
            self.L_hat_left = self.geometry.build_interaction_matrix(feature_hat_left)
            self.L_hat_right = self.geometry.build_interaction_matrix(feature_hat_right)

        elif not self.use_3d_matrix_feature and self.use_delta_matrix:
            self.L_hat_left = self.geometry.build_interaction_matrix_delta(feature_hat_left, last_delta)
            self.L_hat_right = self.geometry.build_interaction_matrix_delta(feature_hat_right, last_delta)

        elif not self.use_3d_matrix_feature and not self.use_delta_matrix:
            self.L_hat_left = self.geometry.build_interaction_matrix(feature_hat_left, last_depth)
            self.L_hat_right = self.geometry.build_interaction_matrix(feature_hat_right, last_depth)

        return self.L_hat_left, self.L_hat_right

    # =========================================================
    def compute_Ldot(self, feature_hat_left, feature_hat_right, nu_B_hat, last_depth=None, last_delta=None,):
        if self.L_hat_left is None or self.L_hat_right is None or nu_B_hat is None:
            self.L_dot_left = None
            self.L_dot_right = None
            return None, None

        if (not self.use_3d_matrix_feature) and self.use_delta_matrix:
            if last_delta is None:
                self.L_dot_left = None
                self.L_dot_right = None
                return None, None

        if (not self.use_3d_matrix_feature) and (not self.use_delta_matrix):
            if last_depth is None:
                self.L_hat_left = None
                self.L_hat_right = None
                return None, None
            
        feature_hat_left = np.asarray(feature_hat_left, dtype=float).flatten()
        feature_hat_right = np.asarray(feature_hat_right, dtype=float).flatten()
        nu_B_hat = np.asarray(nu_B_hat, dtype=float).reshape(6)

        camera_nu_left = self.geometry.T_bc_0 @ nu_B_hat.reshape(6, 1)
        camera_nu_right = self.geometry.T_bc_1 @ nu_B_hat.reshape(6, 1)
        feature_dot_left = self.L_hat_left @ camera_nu_left
        feature_dot_right = self.L_hat_right @ camera_nu_right

        Ldot_rows_left = []
        Ldot_rows_right = []

        if self.use_3d_matrix_feature and self.use_delta_matrix:
            for i in range(self.N):
                idx = 3 * i
                x_l, y_l, delta_l = feature_hat_left[idx:idx + 3]
                x_r, y_r, delta_r = feature_hat_right[idx:idx + 3]
                x_dot_l = feature_dot_left[idx, 0]
                x_dot_r = feature_dot_right[idx, 0]

                y_dot_l = feature_dot_left[idx + 1, 0]
                y_dot_r = feature_dot_right[idx + 1, 0]

                delta_dot_l = feature_dot_left[idx + 2, 0]
                delta_dot_r = feature_dot_right[idx + 2, 0]

                Ldot_rows_left.append(self.geometry.interaction_matrix_delta_3d_dot(x_l, y_l, delta_l, x_dot_l, y_dot_l, delta_dot_l))
                Ldot_rows_right.append(self.geometry.interaction_matrix_delta_3d_dot(x_r, y_r, delta_r, x_dot_r, y_dot_r, delta_dot_r))

        elif self.use_3d_matrix_feature and not self.use_delta_matrix:
            for i in range(self.N):
                idx = 3 * i
                x_l, y_l, Z_l = feature_hat_left[idx:idx + 3]
                x_r, y_r, Z_r = feature_hat_right[idx:idx + 3]

                x_dot_l = feature_dot_left[idx, 0]
                x_dot_r = feature_dot_right[idx, 0]

                y_dot_l = feature_dot_left[idx + 1, 0]
                y_dot_r = feature_dot_right[idx + 1, 0]

                Z_dot_l = feature_dot_left[idx + 2, 0]
                Z_dot_r = feature_dot_right[idx + 2, 0]

                Ldot_rows_left.append(self.geometry.interaction_matrix_3d_dot(x_l, y_l, Z_l, x_dot_l, y_dot_l, Z_dot_l))
                Ldot_rows_right.append(self.geometry.interaction_matrix_3d_dot(x_r, y_r, Z_r, x_dot_r, y_dot_r, Z_dot_r))

        elif not self.use_3d_matrix_feature and self.use_delta_matrix:
            last_delta = np.asarray(last_delta, dtype=float).reshape(self.N)
            for i in range(self.N):
                idx = 2 * i
                x_l, y_l = feature_hat_left[idx:idx + 2]
                x_r, y_r = feature_hat_right[idx:idx + 2]

                delta = last_delta[i]

                x_dot_l = feature_dot_left[idx, 0]
                x_dot_r = feature_dot_right[idx, 0]
                
                y_dot_l = feature_dot_left[idx + 1, 0]
                y_dot_r = feature_dot_right[idx + 1, 0]

                L3_l = self.geometry.interaction_matrix_delta_3d(x_l, y_l, delta)
                delta_dot_l = float(L3_l[2, :] @ camera_nu_left.flatten())

                L3_r = self.geometry.interaction_matrix_delta_3d(x_r, y_r, delta)
                delta_dot_r = float(L3_r[2, :] @ camera_nu_right.flatten())

                Ldot_rows_left.append(self.geometry.interaction_matrix_delta_2d_dot(x_l, y_l, delta, x_dot_l, y_dot_l, delta_dot_l))
                Ldot_rows_right.append(self.geometry.interaction_matrix_delta_2d_dot(x_r, y_r, delta, x_dot_r, y_dot_r, delta_dot_r))

        elif not self.use_3d_matrix_feature and not self.use_delta_matrix:
            last_depth = np.asarray(last_depth, dtype=float).reshape(self.N)
            for i in range(self.N):
                idx = 2 * i
                x_l, y_l = feature_hat_left[idx:idx + 2]
                x_r, y_r = feature_hat_right[idx:idx + 2]

                Z = last_depth[i]

                x_dot_l = feature_dot_left[idx, 0]
                x_dot_r = feature_dot_right[idx, 0]
                
                y_dot_l = feature_dot_left[idx + 1, 0]
                y_dot_r = feature_dot_right[idx + 1, 0]

                L3_l = self.geometry.interaction_matrix_3d(x_l, y_l, Z)
                Z_dot_l = float(L3_l[2, :] @ camera_nu_left.flatten())

                L3_r = self.geometry.interaction_matrix_3d(x_r, y_r, Z)
                Z_dot_r = float(L3_r[2, :] @ camera_nu_right.flatten())

                Ldot_rows_left.append(self.geometry.interaction_matrix_2d_dot(x_l, y_l, Z, x_dot_l, y_dot_l, Z_dot_l))
                Ldot_rows_right.append(self.geometry.interaction_matrix_2d_dot(x_r, y_r, Z, x_dot_r, y_dot_r, Z_dot_r))

        self.L_dot_left = np.vstack(Ldot_rows_left)
        self.L_dot_right = np.vstack(Ldot_rows_right)

        return self.L_dot_left, self.L_dot_right

    # =========================================================
    def compute_control_tau_stereo(self, last_depth, last_delta, nu_B_hat, distance, 
            feature_hat_left, e_norm_left, e_pixel_left, 
            feature_hat_right, e_norm_right, e_pixel_right, 
            dt, tag_lost=False,):
        _ = distance
        self.L_hat_left, self.L_hat_right = self.update_L_hat(feature_hat_left, feature_hat_right, last_depth, last_delta, tag_lost)
        if self.L_hat_left is None or self.L_hat_right is None:
            return np.zeros(6)

        self.L_dot_left, self.L_dot_right = self.compute_Ldot(feature_hat_left, feature_hat_right, nu_B_hat, last_depth, last_delta)
        if self.L_dot_left is None or self.L_dot_right is None:
            return np.zeros(6)

        nu_B_hat = np.asarray(nu_B_hat, dtype=float).reshape(6, 1)

        e_norm_left = np.asarray(e_norm_left, dtype=float).reshape(-1, 1)
        e_norm_right = np.asarray(e_norm_right, dtype=float).reshape(-1, 1)
        e_pixel_left = np.asarray(e_pixel_left, dtype=float).reshape(-1, 1)
        e_pixel_right = np.asarray(e_pixel_right, dtype=float).reshape(-1, 1)

        e_norm = np.vstack((e_norm_left, e_norm_right))
        e_pixel = np.vstack((e_pixel_left, e_pixel_right))

        L_left = self.L_hat_left @ self.geometry.T_bc_0
        L_right = self.L_hat_right @ self.geometry.T_bc_1

        L_left_dot = self.L_dot_left @ self.geometry.T_bc_0
        L_right_dot = self.L_dot_right @ self.geometry.T_bc_1

        L_so = np.vstack((L_left, L_right))
        L_so_dot = np.vstack((L_left_dot, L_right_dot))

        e_dot_hat = L_so @ nu_B_hat

        alpha = self.compute_alpha_so(L_so)
        alpha_inv = np.linalg.pinv(alpha)

        gamma = self.compute_gamma(nu_B_hat)

        l_dot = L_so_dot @ nu_B_hat      

        tau_P = -self.Kp @ alpha_inv @ e_pixel
        tau_D = -self.Kd @ alpha_inv @ e_dot_hat
        tau_L = -alpha_inv @ l_dot
        tau_gamma = alpha_inv @ alpha @ gamma

        tau = tau_P + tau_D + tau_L + tau_gamma

        if np.max(np.abs(e_pixel)) < dead_band:
            tau[:] = 0.0

        self.limit_force(tau)

        return tau.reshape(6)

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
