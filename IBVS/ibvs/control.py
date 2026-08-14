import numpy as np
from .constants import *

class IBVS_Controller:
    def __init__(self, nu_B, distance, e_norm, e_pixel, dt):
        self. sdad

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
            [R_CB,           -R_CB @ S],
            [np.zeros((3,3)),     R_CB]])

        return T_bc
    
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

        return self.L_dot    # =========================================================
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

        return self.L_dot    # =========================================================
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

        return self.L_dot    # =========================================================
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
        Vb = R_CB @ v + np.cross(Wb, P_BC.reshape(3))

        Vb[0] = Vb[0]
        Vb[1] = Vb[1]
        Vb[2] = Vb[2]
        Wb[0] = Wb[0]
        Wb[1] = Wb[1]
        Wb[2] = Wb[2]

        return Vb, Wb
