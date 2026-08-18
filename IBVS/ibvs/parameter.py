import numpy as np

R_BN = np.eye(3)
R_imu = np.diag([
    1.272148604270818**2,
    1.272148604270818**2,
    1.272148604270818**2,
    0.0011478924062028428**2,
    0.0011478924062028428**2,
    0.0011478924062028428**2
])

dead_band = 5
lambda_gain = 0.1
mu = 0.2

Z_DES = 1.5

ROLL_DES_DEG = 0
PITCH_DES_DEG = 0
YAW_DES_DEG = 0

TAG_SIZE = 0.2065
PATCH = 2

# --- Camera Matrix (Intrinsics)
FX = 422.639
FY = 424.172
CX = 426.330
CY = 240.828

# --- Distortion Coefficients (k1, k2, p1, p2, k3) ---
DIST_COEFFS = [-0.0131761, 0.000339851, -0.00139667, 0.00330581, 0.0]

bline = 0.096389

# --- Camera measurement noise ---
sigma_u_px = 1.0
sigma_v_px = 1.0
sigma_Z_m = 0.005
camera_noise_rng = np.random.default_rng(42)

# ROV Properties
m = 29.0
Ixx = 0.492558 + 0.16
Iyy = 0.758506 + 0.30
Izz = 0.919455 + 0.30

Dlin = np.diag([4.0, 4.0, 5.0, 0.0, 0.0, 0.8])

Dquad = np.diag([6.0, 6.0, 8.0, 0.4, 1.19, 0.482])

M = np.diag([m, m, m, Ixx, Iyy, Izz])

# --- SUPRI Physical Offsets (IMU as 0,0 to Cam) ---
P_IC = np.array([
    0.19661243,
    0.06773711,
   -0.00951116])

# --- Transformation Matrix ---
# Camera X Right | Y Down  | Z Front
# MavROS X Front | Y Left  | Z Up
# U-ROVs X Front | Y Right | Z Down

# Camera (OpenCV) -> IMU (FLU)
R_CI = np.array([
    [ 0.02514123, -0.08710579,  0.99588177],
    [-0.99910152,  0.03181017,  0.02800482],
    [-0.03411855, -0.99569106, -0.08622778]], dtype=float)

# IMU (FLU) -> Body (NED)
R_IB = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]], dtype=float)

R_CB = R_IB @ R_CI

P_BC = R_IB @ P_IC


# --- Velocity Limits ---
MAX_LIN_VEL = 1.0
MAX_ANG_VEL = 0.3

MAX_FORCE = 10
MAX_MOMEN = 10


# --- QGC Port Configuration ---
QGC_IP = "192.168.2.1"
QGC_PORT = 5600

stream_w = 848
stream_h = 480

class IBVS_Geometry:
    def __init__(self, N=4, use_3d_matrix_feature=True,):
        self.N = N
        self.use_3d_matrix_feature = use_3d_matrix_feature

        self.T_bc = self.camera_body_adjoint()

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
    def build_interaction_matrix(self, state, depth=None):
        state = np.asarray(state, dtype=float).flatten()
        rows = []

        if self.use_3d_matrix_feature:
            for i in range(self.N):
                idx = 3 * i
                x, y, Z = state[idx:idx + 3]
                rows.append(self.interaction_matrix_3d(x, y, Z))
        else:
            if depth is None:
                return None

            depth = np.asarray(depth, dtype=float).reshape(self.N)
            for i in range(self.N):
                idx = 2 * i
                x, y = state[idx:idx + 2]
                rows.append(self.interaction_matrix_2d(x, y, depth[i]))

        return np.vstack(rows)

    # =========================================================
    def interaction_matrix_delta_2d(self, x, y, delta):

        return np.array([
            [-delta/bline,       0,          delta * x/bline,    x*y,   -(1 + x*x),  y],
            [       0,     -delta/bline,     delta * y/bline,  1 + y*y,    -x*y,    -x]
        ])

    # =========================================================
    def interaction_matrix_delta_3d(self, x, y, delta):

        return np.array([
            [-delta/bline,       0,          delta * x/bline,    x*y,   -(1 + x*x),  y],
            [     0,       -delta/bline,     delta * y/bline,  1 + y*y,    -x*y,    -x],
            [     0,             0,          -delta / bline,     -y,         x,      0]
        ])

    # =========================================================
    def build_interaction_matrix_delta(self, state, deltas=None):
        state = np.asarray(state, dtype=float).flatten()
        rows = []

        if self.use_3d_matrix_feature:
            for i in range(self.N):
                idx = 3 * i
                x, y, deltas = state[idx:idx + 3]
                rows.append(self.interaction_matrix_delta_3d(x, y, deltas))
        else:
            if deltas is None:
                return None

            deltas = np.asarray(deltas, dtype=float).reshape(self.N)
            for i in range(self.N):
                idx = 2 * i
                x, y = state[idx:idx + 2]
                rows.append(self.interaction_matrix_delta_2d(x, y, deltas[i]))

        return np.vstack(rows)

    # =========================================================
    def interaction_matrix_2d_dot(self, x, y, Z, x_dot, y_dot, Z_dot):
        Z2 = Z * Z

        return np.array([
            [Z_dot / Z2,     0.0,    x_dot / Z - x * Z_dot / Z2, x_dot * y + x * y_dot,     -2.0 * x * x_dot,      y_dot],
            [    0.0,    Z_dot / Z2, y_dot / Z - y * Z_dot / Z2, 2.0 * y * y_dot,       -(x_dot * y + x * y_dot), -x_dot]
        ])
    
    # =========================================================
    def interaction_matrix_3d_dot(self, x, y, Z, x_dot, y_dot, Z_dot):
        Z2 = Z * Z

        return np.array([
            [Z_dot / Z2,     0.0,    x_dot / Z - x * Z_dot / Z2, x_dot * y + x * y_dot,     -2.0 * x * x_dot,      y_dot],
            [    0.0,    Z_dot / Z2, y_dot / Z - y * Z_dot / Z2, 2.0 * y * y_dot,       -(x_dot * y + x * y_dot), -x_dot],
            [    0.0,        0.0,            Z_dot / Z2,              -y_dot,                    x_dot,              0.0]
        ])

    # =========================================================
    def interaction_matrix_delta_2d_dot(self, x, y, delta, x_dot, y_dot, delta_dot):
        b = bline

        return np.array([
            [-delta_dot / b,     0.0,        (delta_dot * x + delta * x_dot) / b, x_dot * y + x * y_dot,     -2.0 * x * x_dot,      y_dot],
            [      0.0,      -delta_dot / b, (delta_dot * y + delta * y_dot) / b,    2.0 * y * y_dot,    -(x_dot * y + x * y_dot), -x_dot]
        ])
    
    # =========================================================
    def interaction_matrix_delta_3d_dot(self, x, y, delta, x_dot, y_dot, delta_dot):
        b = bline

        return np.array([
            [-delta_dot / b,     0.0,        (delta_dot * x + delta * x_dot) / b, x_dot * y + x * y_dot,     -2.0 * x * x_dot,      y_dot],
            [      0.0,      -delta_dot / b, (delta_dot * y + delta * y_dot) / b,    2.0 * y * y_dot,    -(x_dot * y + x * y_dot), -x_dot],
            [      0.0,          0.0,                -delta_dot / b,                     y_dot,                   x_dot,              0.0]
        ])

class Shared_State:
    def __init__(self):
        self.tag_lost = True
        self.ukf_initialized = False
        self.last_depth = None
        self.last_delta = None
        self.camera_measurement_valid = False
