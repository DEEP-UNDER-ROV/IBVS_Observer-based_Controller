import numpy as np
from .parameter import *

class UKF_Estimator:
    def __init__(self, shared, nx, feature_dim, N=4,
        use_3d_matrix_feature=True,
        use_delta_matrix=False,
        stereo_cam=None,
        logger=None,
    ):
        self.ukf_state = nx
        self.n_dim = feature_dim
        self.N = N
        self.use_3d_matrix_feature = use_3d_matrix_feature
        self.use_delta_matrix = use_delta_matrix
        self.stereo_cam = stereo_cam

        # ROS-independent logging hook. The node can pass get_logger().
        self.logger = logger
        self.shared = shared
        
        self.geometry = IBVS_Geometry(N=self.N, use_3d_matrix_feature=self.use_3d_matrix_feature,)
        # self.shared = Shared_State()

        self.T_bc_0 = self.geometry.T_bc_0
        self.T_bc_1 = self.geometry.T_bc_1
        self.Minv = np.linalg.inv(M)
        
        # Process and Measurement noise matrices (Initialize as needed)
        # self.Q = np.eye(nx, dtype=np.float64) * 0.001
        self.R_imu = np.diag([
            1.272148604270818**2,
            1.272148604270818**2,
            1.272148604270818**2,
            0.0011478924062028428**2,
            0.0011478924062028428**2,
            0.0011478924062028428**2
        ])
        self.R_camera = np.eye(self.n_dim, dtype=np.float64) * 1e-4

        self.sigma_u_px = float(0.334788)
        self.sigma_v_px = float(0.363501)
        self.sigma_Z = float(0.01)

        self.sigma_x = self.sigma_u_px / FX
        self.sigma_y = self.sigma_v_px / FY

        # Build R_camera for [u, v, Z] measurements.
        self.R_camera = np.zeros((self.n_dim, self.n_dim),dtype=np.float64)
        R_left = np.zeros((self.n_dim, self.n_dim), dtype=np.float64)

        if self.use_3d_matrix_feature:
            for i in range(self.N):
                idx = 3 * i
                R_left[idx, idx] = self.sigma_u_px**2
                R_left[idx + 1, idx + 1] = self.sigma_v_px**2
                R_left[idx + 2, idx + 2] = self.sigma_Z**2

        else:
            for i in range(self.N):
                idx = 2 * i
                R_left[idx, idx] = self.sigma_u_px**2
                R_left[idx + 1, idx + 1] = self.sigma_v_px**2

        # Put each camera covariance into the stereo covariance
        self.R_camera[:self.n_dim, :self.n_dim] = R_left


        self.idx_s = slice(0, self.n_dim)
        self.idx_vB = slice(self.n_dim, self.n_dim + 3)
        self.idx_wB = slice(self.n_dim + 3, self.n_dim + 6)
        self.idx_bo = slice(self.n_dim + 6, self.n_dim + 9)
        self.idx_bg = slice(self.n_dim + 9, self.n_dim + 12)
        self.idx_aB = slice(self.n_dim + 12, self.n_dim + 15)
        self.idx_ba = slice(self.n_dim + 15, self.n_dim + 18)
        

        self.idx_nuB = np.concatenate([
            np.arange(self.idx_vB.start, self.idx_vB.stop),
            np.arange(self.idx_wB.start, self.idx_wB.stop)
        ])

        self.alpha = 0.5
        self.beta = 2.0
        self.kappa = 0.0

        self.q_u = 1e-6
        self.q_v = 1e-6
        self.q_Z = 1e-5

        self.q_vB = 1e-3
        self.q_wB = 1e-3

        self.q_aB = 1e-5

        self.sigma_bg_rw = 7.692845772051322e-7
        self.sigma_ba_rw = 0.003597694747410243
        self.sigma_bo_rw = 1e-4

        self.lambda_ = self.alpha**2 * (self.ukf_state + self.kappa) - self.ukf_state
        self.gamma = np.sqrt(self.ukf_state + self.lambda_)
        self.n_sigma = 2 * self.ukf_state + 1

        self.Wm = np.full(2*self.ukf_state+1,
                          1.0/(2*(self.ukf_state+self.lambda_)))

        self.Wc = np.full(2*self.ukf_state+1,
                          1.0/(2*(self.ukf_state+self.lambda_)))

        self.Wm[0] = self.lambda_/(self.ukf_state+self.lambda_)
        self.Wc[0] = self.Wm[0] + (1-self.alpha**2+self.beta)

        self.camera_imu_timeshift = 0.00702

        self.ukf_x = np.zeros((self.ukf_state,1))
        self.ukf_P = np.eye(self.ukf_state)*1e-3

    # =========================================================
    def _log_info(self, text):
        if self.logger is not None:
            self.logger.info(text)

    # =========================================================
    def pack_state(self, s_C, v_B, w_B, b_o, b_g, a_B, b_a): 
        state = np.zeros(self.ukf_state, dtype=np.float64)

        state[self.idx_s] = np.asarray(s_C, dtype=np.float64).reshape(-1)
        state[self.idx_vB] = np.asarray(v_B, dtype=np.float64).reshape(-1)
        state[self.idx_wB] = np.asarray(w_B, dtype=np.float64).reshape(-1)
        state[self.idx_bo] = np.asarray(b_o, dtype=np.float64).reshape(-1)
        state[self.idx_bg] = np.asarray(b_g, dtype=np.float64).reshape(-1)
        state[self.idx_aB] = np.asarray(a_B, dtype=np.float64).reshape(-1)
        state[self.idx_ba] = np.asarray(b_a, dtype=np.float64).reshape(-1)
        
        return state
    
    # =========================================================
    def unpack_state(self, state):
        state = np.asarray(state, dtype=np.float64).reshape(-1)

        s_C = state[self.idx_s].copy()
        v_B = state[self.idx_vB].copy()
        w_B = state[self.idx_wB].copy()
        b_o = state[self.idx_bo].copy()
        b_g = state[self.idx_bg].copy()
        a_B = state[self.idx_aB].copy()
        b_a = state[self.idx_ba].copy()
        
        return s_C, v_B, w_B, b_o, b_g, a_B, b_a

    # =========================================================
    def reset(self):
        self.ukf_x = np.zeros((self.ukf_state, 1), dtype=np.float64)

        P0 = np.zeros((self.ukf_state, self.ukf_state), dtype=np.float64)
        if self.use_3d_matrix_feature:
            P0[self.idx_s, self.idx_s] = (np.eye(self.n_dim) * 1e-3)
        else:
            P0[self.idx_s, self.idx_s] = (np.eye(self.n_dim) * 1e-3)
        P0[self.idx_vB, self.idx_vB] = (np.eye(3) * 1e-2)
        P0[self.idx_wB, self.idx_wB] = (np.eye(3) * 1e-2)
        P0[self.idx_bo, self.idx_bo] = (np.eye(3) * 1e-2)
        P0[self.idx_bg, self.idx_bg] = (np.eye(3) * 1e-4)
        P0[self.idx_aB, self.idx_aB] = (np.eye(3) * 1e-4)
        P0[self.idx_ba, self.idx_ba] = (np.eye(3) * 1e-2)
     
        self.ukf_P = P0
    
    # =========================================================
    def fossen_acceleration(self, nu_B, tau):
        nu_B = np.asarray(nu_B, dtype=np.float64).reshape(6)
        tau = np.asarray(tau, dtype=np.float64).reshape(6)

        gamma = self.compute_gamma(nu_B.reshape(-1,1))
        nu_dot_B = self.Minv @ (tau + gamma)

        return nu_dot_B
    
    # =========================================================
    def build_Q(self, dt):
        Q = np.zeros((self.ukf_state, self.ukf_state), dtype=np.float64)

        if self.use_3d_matrix_feature:
            q_Sc = np.array([self.q_u, self.q_v, self.q_Z],dtype=np.float64)
        else:
            q_Sc = np.array([self.q_u, self.q_v],dtype=np.float64)

        q_Sc = np.tile(q_Sc, self.N)

        if self.shared.tag_lost:
            adaptive_q_vB = self.q_vB * 1e-4  # Heavily dampen velocity uncertainty growth
            adaptive_q_wB = self.q_wB * 1e-4
        else:
            adaptive_q_vB = self.q_vB
            adaptive_q_wB = self.q_wB

        Q[self.idx_s, self.idx_s] = np.diag(q_Sc)
        Q[self.idx_vB, self.idx_vB] = (np.eye(3) * adaptive_q_vB)
        Q[self.idx_wB, self.idx_wB] = (np.eye(3) * adaptive_q_wB)
        Q[self.idx_bo, self.idx_bo] = (np.eye(3) * self.sigma_bo_rw**2 * dt)
        Q[self.idx_bg, self.idx_bg] = (np.eye(3) * self.sigma_bg_rw**2 * dt)
        Q[self.idx_aB, self.idx_aB] = (np.eye(3) * self.q_aB)
        Q[self.idx_ba, self.idx_ba] = (np.eye(3) * self.sigma_ba_rw**2 * dt)

        return Q

    # =========================================================
    def initialize_ukf_from_camera(self, z_camera):
        z_camera = np.asarray(z_camera, dtype=np.float64).reshape(self.n_dim)

        s_C0 = z_camera.copy()
        v_B0 = np.zeros(3)
        w_B0 = np.zeros(3)
        b_o0 = np.zeros(3)
        b_g0 = np.zeros(3)
        a_B0 = np.zeros(3)
        b_a0 = np.zeros(3)

        self.ukf_x = self.pack_state(s_C0, v_B0, w_B0, b_o0, b_g0, a_B0, b_a0)

        P0 = np.zeros((self.ukf_state, self.ukf_state), dtype=np.float64)
        if self.use_3d_matrix_feature:
            P0[self.idx_s, self.idx_s] = (np.eye(self.n_dim) * 1e-3)
        else:
            P0[self.idx_s, self.idx_s] = (np.eye(self.n_dim) * 1e-3)
        P0[self.idx_vB, self.idx_vB] = (np.eye(3) * 1e-2)
        P0[self.idx_wB, self.idx_wB] = (np.eye(3) * 1e-2)
        P0[self.idx_bo, self.idx_bo] = (np.eye(3) * 1e-2)
        P0[self.idx_bg, self.idx_bg] = (np.eye(3) * 1e-4)
        P0[self.idx_aB, self.idx_aB] = (np.eye(3) * 1e-4)
        P0[self.idx_ba, self.idx_ba] = (np.eye(3) * 1e-2)  
        
        self.ukf_P = P0
        self.shared.ukf_initialized = True
        self._log_info("UKF initialized from stereo camera measurement.")

    # =========================================================
    def generate_sigma_points(self, x, P):
        x = np.asarray(x, dtype=np.float64).reshape(self.ukf_state)
        P = np.asarray(P, dtype=np.float64)
        P = 0.5 * (P + P.T)
        jitter = 1e-9 * np.eye(self.ukf_state)

        try:
            S = np.linalg.cholesky(self.gamma * (P + jitter))

        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals = np.maximum(eigvals, 1e-12)
            P_fixed = (eigvecs @ np.diag(eigvals) @ eigvecs.T)
            S = np.linalg.cholesky(self.gamma * P_fixed)

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
        S = 0.5 * (S + S.T)
        S += 1e-10 * np.eye(measurement_dim)

        return S

    # =========================================================
    def cross_covariance(self, sigma_points, x_mean, z_sigma, z_mean):
        measurement_dim = z_sigma.shape[1]
        Pxz = np.zeros((self.ukf_state, measurement_dim), dtype=np.float64)

        for i in range(self.n_sigma):
            dx = sigma_points[i] - x_mean
            dz = z_sigma[i] - z_mean
            Pxz += self.Wc[i] * np.outer(dx, dz)

        return Pxz

    # =========================================================
    def ukf_process_model(self, state, dt, last_distance=None, tau=None,):
        state = np.asarray(state, dtype=np.float64).reshape(self.ukf_state)
        # last_depth = np.asarray(last_depth, dtype=np.float64).reshape(self.N)
        # last_delta = np.asarray(last_delta, dtype=np.float64).reshape(self.N)

        s_C, v_B, w_B, b_o, b_g, a_B, b_a = self.unpack_state(state)

        if self.shared.tag_lost:
            sC_next = s_C.copy()

        nu_B = np.concatenate([v_B, w_B])
        nu_C = self.T_bc_0 @ nu_B

        if self.use_3d_matrix_feature and self.use_delta_matrix:
            L_sigma = self.geometry.build_interaction_matrix_delta(s_C)
        elif self.use_3d_matrix_feature and not self.use_delta_matrix:
            L_sigma = self.geometry.build_interaction_matrix(s_C)
        elif not self.use_3d_matrix_feature and self.use_delta_matrix:
            L_sigma = self.geometry.build_interaction_matrix_delta(s_C, last_distance)
        elif not self.use_3d_matrix_feature and not self.use_delta_matrix:
            L_sigma = self.geometry.build_interaction_matrix(s_C, last_distance)

        sC_next = s_C + dt * (L_sigma @ nu_C).reshape(-1)
        vB_next = v_B + dt * a_B
        wB_next = w_B + dt * b_o

        bo_next = b_o
        bg_next = b_g
        aB_next = a_B
        ba_next = b_a

        return self.pack_state(sC_next, vB_next, wB_next, bo_next, bg_next, aB_next, ba_next)

    # =========================================================
    def ukf_predict(self, x, P, dt, last_distance=None,):
        sigma = self.generate_sigma_points(x, P)
        sigma_pred = np.zeros_like(sigma)

        for i in range(self.n_sigma):
            sigma_pred[i] = (self.ukf_process_model(sigma[i], dt, last_distance,))

        x_pred = self.weighted_mean(sigma_pred)
        P_pred = self.state_covariance(sigma_pred, x_pred)

        P_pred += self.build_Q(dt)
        P_pred = 0.5 * (P_pred + P_pred.T)
        P_pred += (1e-10 * np.eye(self.ukf_state))

        return (x_pred, P_pred, sigma_pred)
    
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
        P_update = 0.5 * (P_update + P_update.T)
        P_update += 1e-10 * np.eye(self.ukf_state)

        return (x_update, P_update, innovation, S, K, z_mean)

    # =========================================================
    def imu_fcu_measurement_model(self, state, R_NB):
        state = np.asarray(state, dtype=np.float64).reshape(self.ukf_state)

        s_C, v_B, w_B, b_o, b_g, a_B, b_a = self.unpack_state (state)
        
        g_N = np.array([0.0, 0.0, 9.80665],dtype=np.float64)
        g_B = R_NB.T @ g_N

        a_pred_B = (a_B + np.cross(w_B, v_B) - g_B + b_a)
        omega_pred_B = (w_B + b_g)

        z_pred = np.concatenate([a_pred_B, omega_pred_B])

        return z_pred

    # =========================================================
    def predict_imu_fcu_sigma_point(self, sigma_points, R_NB):
        z_sigma = np.zeros((self.n_sigma, 6), dtype=np.float64)

        for i in range(self.n_sigma):
            z_sigma[i] = (self.imu_fcu_measurement_model( sigma_points[i], R_NB))

        return z_sigma

    # =========================================================
    def ukf_update_imu_fcu(self, x_pred, P_pred, sigma_pred, z_imu, R_NB):
        z_imu = np.asarray(z_imu, dtype=np.float64).reshape(6)
        z_sigma = self.predict_imu_fcu_sigma_point(sigma_pred, R_NB)
        z_mean = self.weighted_mean(z_sigma)

        innovation = z_imu - z_mean
        S_standard = self.measurement_covariance(z_sigma, z_mean, self.R_imu)

        mahalanobis_dist = innovation.T @ np.linalg.inv(S_standard) @ innovation

        gamma = 12.59

        if mahalanobis_dist > gamma:
            scale_factor = mahalanobis_dist / gamma
            S = self.measurement_covariance(z_sigma, z_mean, self.R_imu * scale_factor)
        else:
            S = S_standard

        Pxz = self.cross_covariance(sigma_pred, x_pred, z_sigma, z_mean)
        K = np.linalg.solve(S, Pxz.T).T

        x_update = x_pred + K @ innovation
        P_update = P_pred - K @ S @ K.T
        P_update = 0.5 * (P_update + P_update.T)
        P_update += 1e-10 * np.eye(self.ukf_state)

        return x_update, P_update, innovation, S, K, z_mean
