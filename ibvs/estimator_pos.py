import numpy as np
from .parameter import *

class UKF_Estimator:
    def __init__(self, shared, nx, feature_dim, N=4,
        matrix_3d=True,
        matrix_delta=False,
        stereo_cam=None,
        use_fossen=False,
        logger=None,
        q_fcu_std_deg=5.0,):

        self.ukf_state = nx
        self.n_dim = feature_dim
        self.N = N
        self.matrix_3d = matrix_3d
        self.matrix_delta = matrix_delta
        self.stereo_cam = stereo_cam
        self.ukf_fossen = use_fossen

        self.logger = logger
        self.shared = shared
        
        self.geometry = IBVS_Geometry(N=self.N, matrix_3d=self.matrix_3d,)

        self.Minv = np.linalg.inv(M)

        self.R_camera = np.eye(self.n_dim, dtype=np.float64) * 1e-4        
        self.R_imu_fcu = np.diag([
            0.08995449047568972**2 * 100,
            0.08995449047568972**2 * 100,
            0.08995449047568972**2 * 100,
            (8.116825044985731e-05)**2 * 100,
            (8.116825044985731e-05)**2 * 100,
            (8.116825044985731e-05)**2 * 100
        ])
        self.R_imu_cam = np.diag([
            0.015734928187827825**2,
            0.015734928187827825**2,
            0.015734928187827825**2,
            0.003471248742761046**2,
            0.003471248742761046**2,
            0.003471248742761046**2
        ])

        sigma_gyro_density_fcu = 8.116825044985731e-05
        dt_q = 0.2
        sigma_q_fcu = sigma_gyro_density_fcu * np.sqrt(1.0 / dt_q) * dt_q
        self.R_q_fcu = np.eye(3) * sigma_q_fcu**2

        # sigma_q_fcu = np.deg2rad(float(q_fcu_std_deg))
        # self.R_q_fcu = np.eye(3, dtype=np.float64) * sigma_q_fcu**2

        self.sigma_u_px = float(0.334788)
        self.sigma_v_px = float(0.363501)
        self.sigma_Z = float(0.01)

        if self.matrix_3d:
            self.R_camera = np.zeros((3 * self.N, 3 * self.N),dtype=np.float64)
            for i in range(self.N):
                idx = 3 * i
                self.R_camera[idx,     idx]     = self.sigma_u_px ** 2
                self.R_camera[idx + 1, idx + 1] = self.sigma_v_px ** 2
                self.R_camera[idx + 2, idx + 2] = self.sigma_Z ** 2

        else:
            self.R_camera = np.zeros((2 * self.N, 2 * self.N),dtype=np.float64)
            for i in range(self.N):
                idx = 2 * i
                self.R_camera[idx,     idx]     = self.sigma_u_px ** 2
                self.R_camera[idx + 1, idx + 1] = self.sigma_v_px ** 2

        self.idx_s    = slice(0, self.n_dim)
        self.idx_qN   = slice(self.n_dim, self.n_dim + 4)
        self.idx_pN   = slice(self.n_dim + 4, self.n_dim + 7)
        self.idx_vB   = slice(self.n_dim + 7, self.n_dim + 10)
        self.idx_wB   = slice(self.n_dim + 10, self.n_dim + 13)
        self.idx_bo   = slice(self.n_dim + 13, self.n_dim + 16)
        self.idx_bg_F = slice(self.n_dim + 16, self.n_dim + 19)
        self.idx_aB   = slice(self.n_dim + 19, self.n_dim + 22)
        self.idx_ba_F = slice(self.n_dim + 22, self.n_dim + 25)
        self.idx_bg_C = slice(self.n_dim + 25, self.n_dim + 28)
        self.idx_ba_C = slice(self.n_dim + 28, self.n_dim + 31)

        self.idx_nuB = np.concatenate([
            np.arange(self.idx_vB.start, self.idx_vB.stop),
            np.arange(self.idx_wB.start, self.idx_wB.stop)
        ])

        self.alpha = 0.3
        self.beta = 2.0
        self.kappa = 0.0

        self.q_u = 1e-6
        self.q_v = 1e-6
        self.q_Z = 1e-5

        self.q_vB = 1e-4
        self.q_wB = 1e-3
        self.q_aB = 1e-5

        self.q_pN = 1e-4
        self.q_qN = 1e-4

        self.sigma_bg_F_rw = 7.692845772051322e-7
        self.sigma_ba_F_rw = 0.003597694747410243
        self.sigma_bo_rw   = 1e-4

        self.sigma_bg_C_rw = 4.076264572369842e-6
        self.sigma_ba_C_rw = 3.5949400613806615e-4

        self.lambda_ = self.alpha**2 * (self.ukf_state + self.kappa) - self.ukf_state
        self.gamma = np.sqrt(self.ukf_state + self.lambda_)
        self.n_sigma = 2 * self.ukf_state + 1

        self.Wm = np.full(2*self.ukf_state+1,
                          1.0/(2*(self.ukf_state+self.lambda_)))

        self.Wc = np.full(2*self.ukf_state+1,
                          1.0/(2*(self.ukf_state+self.lambda_)))

        self.Wm[0] = self.lambda_/(self.ukf_state+self.lambda_)
        self.Wc[0] = self.Wm[0] + (1-self.alpha**2+self.beta)

        self.ukf_x = np.zeros((self.ukf_state,1))
        self.ukf_P = np.eye(self.ukf_state)*1e-3
        self.tau_hat = np.zeros((6,1))
        self.current_alpha_B = None

    @staticmethod
    def _normalize_quaternion(q):
        q = np.asarray(q, dtype=np.float64).reshape(4)
        n = np.linalg.norm(q)
        if not np.isfinite(n) or n < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return q / n

    @staticmethod
    def _quat_multiply(q1, q2):
        x1, y1, z1, w1 = np.asarray(q1, dtype=np.float64).reshape(4)
        x2, y2, z2, w2 = np.asarray(q2, dtype=np.float64).reshape(4)

        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ], dtype=np.float64)

    @staticmethod
    def _quat_conjugate(q):
        q = np.asarray(q, dtype=np.float64).reshape(4)
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)

    @classmethod
    def _quat_inverse(cls, q):
        q = cls._normalize_quaternion(q)
        return cls._quat_conjugate(q)

    @staticmethod
    def _quat_from_rotvec(rotvec):
        rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
        theta = np.linalg.norm(rotvec)

        if theta < 1e-12:
            # First-order approximation is enough for a very small increment.
            q = np.array([
                0.5 * rotvec[0],
                0.5 * rotvec[1],
                0.5 * rotvec[2],
                1.0], dtype=np.float64)
            return q / np.linalg.norm(q)

        axis = rotvec / theta
        half = 0.5 * theta
        s = np.sin(half)

        return np.array([
            axis[0] * s,
            axis[1] * s,
            axis[2] * s,
            np.cos(half)], dtype=np.float64)

    @staticmethod
    def _quat_to_rotmat(q):
        x, y, z, w = UKF_Estimator._normalize_quaternion(q)

        return np.array([
            [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
            [2.0*(x*y + z*w),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)],
            [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x*x + y*y)]], dtype=np.float64)

    @classmethod
    def _quat_to_rotvec(cls, q):
        q = cls._normalize_quaternion(q)
        if q[3] < 0.0:
            q = -q

        v = q[:3]
        v_norm = np.linalg.norm(v)

        if v_norm < 1e-12:
            return 2.0 * v

        angle = 2.0 * np.arctan2(v_norm, np.clip(q[3], -1.0, 1.0))
        return v * (angle / v_norm)

    @classmethod
    def _quat_error_rotvec(cls, q_ref, q_meas):
        q_ref = cls._normalize_quaternion(q_ref)
        q_meas = cls._normalize_quaternion(q_meas)
        q_err = cls._quat_multiply(cls._quat_inverse(q_ref), q_meas)
        return cls._quat_to_rotvec(q_err)

    @classmethod
    def _align_quaternion_sign(cls, q, q_ref):
        q = cls._normalize_quaternion(q)
        q_ref = cls._normalize_quaternion(q_ref)
        return -q if np.dot(q, q_ref) < 0.0 else q

    @classmethod
    def _propagate_quaternion(cls, q, omega_B, dt):
        q = cls._normalize_quaternion(q)
        dq = cls._quat_from_rotvec(np.asarray(omega_B, dtype=np.float64).reshape(3) * float(dt))
        q_next = cls._quat_multiply(q, dq)
        return cls._normalize_quaternion(q_next)

    @staticmethod
    def _rotmat_to_quat(R):
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        if not np.isfinite(R).all():
            raise ValueError("Invalid rotation matrix.")

        trace = np.trace(R)
        if trace > 0.0:
            s = 2.0 * np.sqrt(trace + 1.0)
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s

        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 0.0))
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
            w = (R[2, 1] - R[1, 2]) / s

        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 0.0))
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
            w = (R[0, 2] - R[2, 0]) / s

        else:
            s = 2.0 * np.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 0.0))
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
            w = (R[1, 0] - R[0, 1]) / s

        q = np.array([x, y, z, w], dtype=np.float64)

        return UKF_Estimator._normalize_quaternion(q)
    
    # =========================================================
    def _log_info(self, text):
        if self.logger is not None:
            self.logger.info(text)

    # =========================================================
    def pack_state(self, s_C, p_N, v_B, w_B, b_o, b_gF, a_B, b_aF, b_gC, b_aC, q_N=None,): 
        state = np.zeros(self.ukf_state, dtype=np.float64)

        state[self.idx_s]    = np.asarray(s_C, dtype=np.float64).reshape(-1)
        state[self.idx_pN]   = np.asarray(p_N, dtype=np.float64).reshape(3)
        state[self.idx_vB]   = np.asarray(v_B, dtype=np.float64).reshape(3)
        state[self.idx_wB]   = np.asarray(w_B, dtype=np.float64).reshape(3)
        state[self.idx_bo]   = np.asarray(b_o, dtype=np.float64).reshape(3)
        state[self.idx_bg_F] = np.asarray(b_gF, dtype=np.float64).reshape(3)
        state[self.idx_aB]   = np.asarray(a_B, dtype=np.float64).reshape(3)
        state[self.idx_ba_F] = np.asarray(b_aF, dtype=np.float64).reshape(3)
        state[self.idx_bg_C] = np.asarray(b_gC, dtype=np.float64).reshape(3)
        state[self.idx_ba_C] = np.asarray(b_aC, dtype=np.float64).reshape(3)

        if q_N is None:
            q_N = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        state[self.idx_qN] = self._normalize_quaternion(q_N)
        
        return state
    
    # =========================================================
    def unpack_state(self, state):
        state = np.asarray(state, dtype=np.float64).reshape(-1)

        s_C  = state[self.idx_s].copy()
        q_N  = self._normalize_quaternion(state[self.idx_qN].copy())
        p_N  = state[self.idx_pN].copy()
        v_B  = state[self.idx_vB].copy()
        w_B  = state[self.idx_wB].copy()
        b_o  = state[self.idx_bo].copy()
        b_gF = state[self.idx_bg_F].copy()
        a_B  = state[self.idx_aB].copy()
        b_aF = state[self.idx_ba_F].copy()
        b_gC = state[self.idx_bg_C].copy()
        b_aC = state[self.idx_ba_C].copy()
        
        return s_C, q_N, p_N, v_B, w_B, b_o, b_gF, a_B, b_aF, b_gC, b_aC

    # =========================================================
    def reset(self):
        self.ukf_x = np.zeros((self.ukf_state, 1), dtype=np.float64)
        self.ukf_x[self.idx_qN] = np.array([0.0, 0.0, 0.0, 1.0]).reshape(4, 1)

        P0 = np.zeros((self.ukf_state, self.ukf_state), dtype=np.float64)

        P0[self.idx_s, self.idx_s]       = np.eye(self.n_dim) * 1e-3
        P0[self.idx_qN, self.idx_qN]     = np.eye(4) * 1e-3
        P0[self.idx_pN, self.idx_pN]     = np.eye(3) * 1e-2
        P0[self.idx_vB, self.idx_vB]     = np.eye(3) * 1e-2
        P0[self.idx_wB, self.idx_wB]     = np.eye(3) * 1e-2
        P0[self.idx_bo, self.idx_bo]     = np.eye(3) * 1e-2
        P0[self.idx_bg_F, self.idx_bg_F] = np.eye(3) * 1e-4
        P0[self.idx_aB, self.idx_aB]     = np.eye(3) * 1e-4
        P0[self.idx_ba_F, self.idx_ba_F] = np.eye(3) * 1e-2
        P0[self.idx_bg_C, self.idx_bg_C] = np.eye(3) * 1e-4
        P0[self.idx_ba_C, self.idx_ba_C] = np.eye(3) * 1e-2
     
        self.ukf_P = P0
    
    # =========================================================
    def build_Q(self, dt):
        Q = np.zeros((self.ukf_state, self.ukf_state), dtype=np.float64)

        if self.matrix_3d:
            q_Sc = np.array([self.q_u, self.q_v, self.q_Z],dtype=np.float64)
        else:
            q_Sc = np.array([self.q_u, self.q_v],dtype=np.float64)

        q_Sc = np.tile(q_Sc, self.N)

        adaptive_q_vB = self.q_vB * 0.1 if self.shared.tag_lost else self.q_vB
        adaptive_q_wB = self.q_wB * 0.1 if self.shared.tag_lost else self.q_wB

        Q[self.idx_s, self.idx_s]       = np.diag(q_Sc)
        Q[self.idx_qN, self.idx_qN]     = np.eye(4) * self.q_qN * max(dt, 1e-6)
        Q[self.idx_pN, self.idx_pN]     = np.eye(3) * self.q_pN
        Q[self.idx_vB, self.idx_vB]     = np.eye(3) * adaptive_q_vB
        Q[self.idx_wB, self.idx_wB]     = np.eye(3) * adaptive_q_wB
        Q[self.idx_bo, self.idx_bo]     = np.eye(3) * (self.sigma_bo_rw**2) * dt
        Q[self.idx_bg_F, self.idx_bg_F] = np.eye(3) * (self.sigma_bg_F_rw**2) * dt
        Q[self.idx_aB, self.idx_aB]     = np.eye(3) * self.q_aB
        Q[self.idx_ba_F, self.idx_ba_F] = np.eye(3) * (self.sigma_ba_F_rw**2) * dt
        Q[self.idx_bg_C, self.idx_bg_C] = np.eye(3) * (self.sigma_bg_C_rw**2) * dt
        Q[self.idx_ba_C, self.idx_ba_C] = np.eye(3) * (self.sigma_ba_C_rw**2) * dt

        return Q

    # =========================================================
    def initialize_ukf_from_camera(self, z_camera, q_N0=None):
        z_camera = np.asarray(z_camera, dtype=np.float64).reshape(self.n_dim)
        s_C0 = z_camera.copy()

        q_N0 = self._normalize_quaternion(q_N0) if q_N0 is not None else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.ukf_x = self.pack_state(s_C0, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3),
                                      np.zeros(3), np.zeros(3), np.zeros(3),
                                      np.zeros(3), np.zeros(3), q_N=q_N0)

        P0 = np.zeros((self.ukf_state, self.ukf_state), dtype=np.float64)

        P0[self.idx_s, self.idx_s]       = np.eye(self.n_dim) * 1e-3
        P0[self.idx_qN, self.idx_qN]     = np.eye(4) * 1e-3
        P0[self.idx_pN, self.idx_pN]     = np.eye(3) * 1e-2
        P0[self.idx_vB, self.idx_vB]     = np.eye(3) * 1e-2
        P0[self.idx_wB, self.idx_wB]     = np.eye(3) * 1e-2
        P0[self.idx_bo, self.idx_bo]     = np.eye(3) * 1e-2
        P0[self.idx_bg_F, self.idx_bg_F] = np.eye(3) * 1e-4
        P0[self.idx_aB, self.idx_aB]     = np.eye(3) * 1e-4
        P0[self.idx_ba_F, self.idx_ba_F] = np.eye(3) * 1e-2
        P0[self.idx_bg_C, self.idx_bg_C] = np.eye(3) * 1e-4
        P0[self.idx_ba_C, self.idx_ba_C] = np.eye(3) * 1e-2

        self.ukf_P = P0
        self.shared.ukf_initialized = True
        self._log_info("UKF initialized from stereo camera measurement.")

    # =========================================================
    def generate_sigma_points(self, x, P):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if not np.isfinite(P).all() or not np.isfinite(x).all():
            if self.logger is not None:
                self.logger.error("Resetting covariance.")
            self.reset()
            x = self.ukf_x.reshape(-1)
            P = self.ukf_P

        P = (P + P.T) / 2.0
        max_cov = 1e3
        P = np.clip(P, -max_cov, max_cov)
        P_fixed = P + np.eye(P.shape[0]) * 1e-6
        scaled_P = (self.ukf_state + self.lambda_) * P_fixed

        try:
            S = np.linalg.cholesky(scaled_P)
        except np.linalg.LinAlgError:
            scaled_P = (scaled_P + scaled_P.T) / 2.0
            scaled_P = np.nan_to_num(scaled_P, nan=1e-6, posinf=max_cov, neginf=-max_cov)
            U, Sigma, _ = np.linalg.svd(scaled_P)
            S = U @ np.diag(np.sqrt(np.clip(Sigma, 0.0, None)))

        sigma = np.zeros((self.n_sigma, self.ukf_state), dtype=np.float64)
        sigma[0] = x

        for i in range(self.ukf_state):
            sigma[i + 1] = (x + S[:, i])
            sigma[i + 1 + self.ukf_state] = (x - S[:, i])

        return sigma

    # =========================================================
    def weighted_mean(self, sigma_points):
        sigma_points = np.asarray(sigma_points, dtype=np.float64)

        return np.sum(self.Wm[:, None] * sigma_points, axis=0)
    
    # =========================================================
    def state_covariance(self, sigma_points, x_mean):
        sigma_points = np.asarray(sigma_points, dtype=np.float64)
        x_mean = np.asarray(x_mean, dtype=np.float64).reshape(-1)
        P = np.zeros((self.ukf_state, self.ukf_state), dtype=np.float64)
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
        S = 0.5 * (S + S.T) + 1e-10 * np.eye(measurement_dim)
        return S

    # =========================================================
    def cross_covariance(self, sigma_points, x_mean, z_sigma, z_mean):
        sigma_points = np.asarray(sigma_points, dtype=np.float64)
        x_mean = np.asarray(x_mean, dtype=np.float64).reshape(-1)
        z_sigma = np.asarray(z_sigma, dtype=np.float64)
        z_mean = np.asarray(z_mean, dtype=np.float64).reshape(-1)
        measurement_dim = z_sigma.shape[1]
        Pxz = np.zeros((self.ukf_state, measurement_dim), dtype=np.float64)
        for i in range(self.n_sigma):
            dx = sigma_points[i] - x_mean
            dz = z_sigma[i] - z_mean
            Pxz += self.Wc[i] * np.outer(dx, dz)
        return Pxz
    
    # =========================================================
    def fossen_acceleration(self, nu_B, tau):
        nu_B = np.asarray(nu_B, dtype=np.float64).reshape(6)
        tau = np.asarray(tau, dtype=np.float64).reshape(6)

        gamma = self.geometry.compute_gamma(nu_B.reshape(-1,1))
        gamma = np.asarray(gamma, dtype=np.float64).reshape(6)
        nu_dot_B = self.Minv @ (tau + gamma)

        return np.asarray(nu_dot_B, dtype=np.float64).reshape(6)
    
    # =========================================================
    def ukf_process_model(self, state, dt, last_distance=None, tau=None,):
        state = np.asarray(state, dtype=np.float64).reshape(self.ukf_state)
        s_C, q_N, p_N, v_B, w_B, b_o, b_gF, a_B, b_aF, b_gC, b_aC = self.unpack_state(state)

        qN_next = self._propagate_quaternion(q_N, w_B, dt)
        R_NB = self._quat_to_rotmat(q_N)

        nu_B = np.concatenate([v_B, w_B])
        nu_C = self.geometry.T_bc_0 @ nu_B

        if last_distance is not None:
            last_distance = np.asarray(last_distance, dtype=np.float64).reshape(self.N)

        if self.matrix_3d and self.matrix_delta:
            L_sigma = self.geometry.build_interaction_matrix_delta(s_C)
        elif self.matrix_3d and not self.matrix_delta:
            L_sigma = self.geometry.build_interaction_matrix(s_C)
        elif not self.matrix_3d and self.matrix_delta:
            L_sigma = self.geometry.build_interaction_matrix_delta(s_C, last_distance)
        elif not self.matrix_3d and not self.matrix_delta:
            L_sigma = self.geometry.build_interaction_matrix(s_C, last_distance)

        tau = self.tau_hat if tau is None else tau
        tau = np.asarray(tau, dtype=np.float64).reshape(6)

        sC_next = s_C if self.shared.tag_lost else s_C + dt * (L_sigma @ nu_C).reshape(-1)
        
        if self.ukf_fossen:
            nu_dot_B = self.fossen_acceleration(nu_B, tau)
            a_fossen = nu_dot_B[:3]
            alpha_fossen = nu_dot_B[3:]
            
            alpha_B = alpha_fossen + b_o
            aB_next = a_fossen 
            
        else:
            alpha_B = b_o
            aB_next = a_B

        if self.shared.tag_lost:
            damping_factor = 0.90 
            vB_next = (v_B + dt * aB_next) * damping_factor
            wB_next = (w_B + dt * alpha_B) * damping_factor
        else:
            vB_next = v_B + dt * aB_next
            wB_next = w_B + dt * alpha_B
        
        pN_next = p_N + dt * (R_NB @ v_B)

        return self.pack_state(sC_next, pN_next, vB_next, wB_next, b_o, b_gF, aB_next, b_aF, b_gC, b_aC, qN_next,)
    
    # =========================================================
    def ukf_predict(self, x, P, dt, last_distance=None, tau=None,):
        sigma = self.generate_sigma_points(x, P)
        sigma_pred = np.zeros_like(sigma)

        for i in range(self.n_sigma):
            sigma_pred[i] = self.ukf_process_model(sigma[i], dt, last_distance, tau,)

        x_pred = self.weighted_mean(sigma_pred)
        P_pred = self.state_covariance(sigma_pred, x_pred) + self.build_Q(dt)
        P_pred = 0.5 * (P_pred + P_pred.T) + (1e-10 * np.eye(self.ukf_state))
        x_pred[self.idx_qN] = self._normalize_quaternion(x_pred[self.idx_qN])

        return x_pred, P_pred, sigma_pred
    
    # =========================================================
    def camera_measurement_model(self, state):
        state = np.asarray(state, dtype=np.float64).reshape(self.ukf_state)
        z_camera = state[self.idx_s].copy()

        return z_camera

    # =========================================================
    def predict_camera_sigma_points(self, sigma_points):
        z_sigma = np.zeros((self.n_sigma, self.n_dim), dtype=np.float64)
        for i in range(self.n_sigma):
            z_sigma[i] = self.camera_measurement_model(sigma_points[i])

        return z_sigma

    # =========================================================
    def ukf_update_camera(self, x_pred, P_pred, sigma_pred, z_camera):
        z_camera = np.asarray(z_camera, dtype=np.float64).reshape(self.n_dim)

        z_sigma = self.predict_camera_sigma_points(sigma_pred)
        z_mean = self.weighted_mean(z_sigma)

        S = self.measurement_covariance(z_sigma, z_mean, self.R_camera)
        Pxz = self.cross_covariance(sigma_pred, x_pred, z_sigma, z_mean)

        K = np.linalg.solve(S, Pxz.T).T
        innovation = z_camera - z_mean
        x_update = x_pred + K @ innovation
        P_update = P_pred - K @ S @ K.T
        P_update = 0.5 * (P_update + P_update.T) + 1e-10 * np.eye(self.ukf_state)

        return x_update, P_update, innovation, S, K, z_mean

    # =========================================================
    def imu_fcu_measurement_model(self, state):
        state = np.asarray(state, dtype=np.float64).reshape(self.ukf_state)
        s_C, q_N, p_N, v_B, w_B, b_o, b_gF, a_B, b_aF, b_gC, b_aC = self.unpack_state(state)

        R_NB = self._quat_to_rotmat(q_N)
        g_N = np.array([0.0, 0.0, 9.80665],dtype=np.float64)
        g_B = R_NB.T @ g_N

        R_BF = T_BF[:3, :3]
        P_BF = T_BF[:3, 3]

        a_body_B = a_B + np.cross(w_B, v_B) - g_B
        a_pred_B = a_body_B + (R_BF @ b_aF)
        omega_pred_B = w_B + (R_BF @ b_gF)

        return np.concatenate([a_pred_B, omega_pred_B])

    # =========================================================
    def imu_cam_measurement_model(self, state, ):
        state = np.asarray(state, dtype=np.float64).reshape(self.ukf_state)
        alpha_B = self.current_alpha_B
        s_C, q_N, p_N, v_B, w_B, b_o, b_gF, a_B, b_aF, b_gC, b_aC = self.unpack_state(state)

        if self.ukf_fossen:
            nu_B = np.concatenate([v_B, w_B])
            tau = np.asarray(self.tau_hat, dtype=np.float64).reshape(6)
            nu_dot_B = self.fossen_acceleration(nu_B, tau)
            alpha_fossen = nu_dot_B[3:]
            alpha_B = alpha_fossen + b_o
        else:
            alpha_B = b_o

        R_NB = self._quat_to_rotmat(q_N)
        g_N = np.array([0.0, 0.0, 9.80665], dtype=np.float64)
        g_B = R_NB.T @ g_N

        R_BI = T_BI[:3, :3]
        P_BI = T_BI[:3, 3]

        a_origin = a_B + np.cross(w_B, v_B) - g_B
        lever_arm = np.cross(alpha_B, P_BI) + np.cross(w_B, np.cross(w_B, P_BI))
        a_body = a_origin + lever_arm
    
        a_pred_B = a_body + (R_BI @ b_aC)
        omega_pred_B = w_B + (R_BI @ b_gC)

        return np.concatenate([a_pred_B, omega_pred_B])
    
    # =========================================================
    def predict_imu_fcu_sigma_point(self, sigma_points):
        z_sigma = np.zeros((self.n_sigma, 6), dtype=np.float64)
        for i in range(self.n_sigma):
            z_sigma[i] = self.imu_fcu_measurement_model(sigma_points[i])

        return z_sigma

    # =========================================================
    def predict_imu_cam_sigma_point(self, sigma_points):
        z_sigma = np.zeros((self.n_sigma, 6), dtype=np.float64)
        for i in range(self.n_sigma):
            z_sigma[i] = self.imu_cam_measurement_model(sigma_points[i])

        return z_sigma

    # =========================================================
    def ukf_update_imu_fcu(self, x_pred, P_pred, sigma_pred, z_imu,):
        z_imu = np.asarray(z_imu, dtype=np.float64).reshape(6)
        z_sigma = self.predict_imu_fcu_sigma_point(sigma_pred)
        z_mean = self.weighted_mean(z_sigma)
        innovation = z_imu - z_mean
        S_standard = self.measurement_covariance(z_sigma, z_mean, self.R_imu_fcu)
        mahalanobis_dist = float(innovation.T @ np.linalg.solve(S_standard, innovation))
        gamma = 12.59

        if mahalanobis_dist > gamma:
            scale_factor = mahalanobis_dist / gamma
            S = self.measurement_covariance(z_sigma, z_mean, self.R_imu_fcu * scale_factor)
        else:
            S = S_standard

        Pxz = self.cross_covariance(sigma_pred, x_pred, z_sigma, z_mean)
        K = np.linalg.solve(S, Pxz.T).T

        x_update = x_pred + K @ innovation
        x_update[self.idx_qN] = self._normalize_quaternion(x_update[self.idx_qN])

        P_update = P_pred - K @ S @ K.T
        P_update = 0.5 * (P_update + P_update.T) + 1e-10 * np.eye(self.ukf_state)

        return x_update, P_update, innovation, S, K, z_mean

    # =========================================================
    def ukf_update_imu_cam(self, x_pred, P_pred, sigma_pred, z_imu_cam):
        z_imu_cam = np.asarray(z_imu_cam, dtype=np.float64).reshape(6)
        z_sigma = self.predict_imu_cam_sigma_point(sigma_pred)
        z_mean = self.weighted_mean(z_sigma)
        innovation = z_imu_cam - z_mean

        S_standard = self.measurement_covariance(z_sigma, z_mean, self.R_imu_cam)
        mahalanobis_dist = float(innovation.T @ np.linalg.solve(S_standard, innovation))
        gamma = 12.59

        if mahalanobis_dist > gamma:
            scale_factor = mahalanobis_dist / gamma
            S = self.measurement_covariance(z_sigma, z_mean, self.R_imu_cam * scale_factor)
        else:
            S = S_standard

        Pxz = self.cross_covariance(sigma_pred, x_pred, z_sigma, z_mean)
        K = np.linalg.solve(S, Pxz.T).T

        x_update = x_pred + K @ innovation
        x_update[self.idx_qN] = self._normalize_quaternion(x_update[self.idx_qN])

        P_update = P_pred - K @ S @ K.T
        P_update = 0.5 * (P_update + P_update.T) + 1e-10 * np.eye(self.ukf_state)

        return x_update, P_update, innovation, S, K, z_mean
    
    # =========================================================
    def quaternion_measurement_model(self, state, q_ref):
        state = np.asarray(state, dtype=np.float64).reshape(self.ukf_state)
        q_state = state[self.idx_qN]
        return self._quat_error_rotvec(q_ref, q_state)

    # =========================================================
    def predict_quaternion_sigma_points(self, sigma_points, q_ref):
        z_sigma = np.zeros((self.n_sigma, 3), dtype=np.float64)
        for i in range(self.n_sigma):
            z_sigma[i] = self.quaternion_measurement_model(sigma_points[i],q_ref,)

        return z_sigma

    # =========================================================
    def ukf_update_attitude_fcu(self, x_pred, P_pred, sigma_pred, q_fcu,):
        q_fcu = self._normalize_quaternion(q_fcu)
        q_ref = self._normalize_quaternion(x_pred[self.idx_qN])

        z_sigma = self.predict_quaternion_sigma_points(sigma_pred, q_ref)
        z_mean = self.weighted_mean(z_sigma)
        z_actual = self._quat_error_rotvec(q_ref, q_fcu)
        innovation = z_actual - z_mean

        S = self.measurement_covariance(z_sigma, z_mean, self.R_q_fcu,)
        Pxz = self.cross_covariance(sigma_pred, x_pred, z_sigma, z_mean,)
        K = np.linalg.solve(S, Pxz.T).T

        x_update = x_pred + K @ innovation
        x_update[self.idx_qN] = self._normalize_quaternion(x_update[self.idx_qN])

        P_update = P_pred - K @ S @ K.T
        P_update = 0.5 * (P_update + P_update.T)
        P_update += 1e-10 * np.eye(self.ukf_state)

        return (x_update, P_update, innovation, S, K, z_actual,)