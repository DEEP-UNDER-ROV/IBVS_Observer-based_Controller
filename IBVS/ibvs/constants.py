import numpy as np

# --- Camera Matrix (Intrinsics)
FX = 422.639
FY = 424.172
CX = 426.330
CY = 240.828

# --- Distortion Coefficients (k1, k2, p1, p2, k3) ---
DIST_COEFFS = [-0.0131761, 0.000339851, -0.00139667, 0.00330581, 0.0]

# --- Camera measurement noise ---
sigma_u_px = 1.0
sigma_v_px = 1.0
sigma_Z_m = 0.005

camera_noise_rng = np.random.default_rng(42)

bline = 0.096389

stream_w = 848
stream_h = 480


# --- ASEPP Physical Offsets (Camera relative to CoG) ---
# P_CB_X =  0.17
# P_CB_Y = -0.0475
# P_CB_Z =  0.0


m = 29.0
Ixx = 0.492558 + 0.16
Iyy = 0.758506 + 0.30
Izz = 0.919455 + 0.30

Dlin = np.diag([4.0, 4.0, 5.0, 0.0, 0.0, 0.8])

Dquad = np.diag([6.0, 6.0, 8.0, 0.4, 1.19, 0.482])

M = np.diag([m, m, m, Ixx, Iyy, Izz])

R_BN = np.eye(3)
R_imu = np.diag([
    1.272148604270818**2,
    1.272148604270818**2,
    1.272148604270818**2,
    0.0011478924062028428**2,
    0.0011478924062028428**2,
    0.0011478924062028428**2
])


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


dead_band = 5
lambda_gain = 0.1
mu = 0.2

Z_DES = 1.5

ROLL_DES_DEG = 0
PITCH_DES_DEG = 0
YAW_DES_DEG = 0

TAG_SIZE = 0.2065
PATCH = 2

# ---------------- Gains (TUNE IN WATER) ----------------
K_SURGE = 400
K_SWAY  = 400
K_HEAVE = 400
K_ROLL  = 400
K_PITCH = 400
K_YAW   = 400


# --- Velocity Limits ---
MAX_LIN_VEL = 0.3
MAX_ANG_VEL = 0.3

MAX_FORCE = 10
MAX_MOMEN = 5


# --- QGC Port Configuration ---
QGC_IP = "192.168.2.1"
QGC_PORT = 5600
