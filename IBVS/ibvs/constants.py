import numpy as np
# --- Camera Matrix (Intrinsics)
FX = 422.639
FY = 424.172
CX = 426.330
CY = 240.828


# --- Distortion Coefficients (k1, k2, p1, p2, k3) ---
DIST_COEFFS = [-0.0131761, 0.000339851, -0.00139667, 0.00330581, 0.0]

stream_w = 848
stream_h = 480


# # --- ASEPP Physical Offsets (Camera relative to CoG) ---
# P_CB_X =  0.17
# P_CB_Y = -0.0475
# P_CB_Z =  0.0

# --- SUPRI Physical Offsets (Camera relative to CoG) ---
P_IC = np.array([
    0.19661243,
    0.06773711,
   -0.00951116])

mu = 0.2
Z_DES = 1.5
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
MAX_LIN_VEL = 0.5
MAX_ANG_VEL = 0.2


# --- QGC Port Configuration ---
QGC_IP = "192.168.2.1"
QGC_PORT = 5600


# --- Transformation Matrix ---
# Camera X Right | Y Down  | Z Front
# MavROS X Front | Y Left  | Z Up
# U-ROVs X Front | Y Right | Z Down

# Camera (OpenCV) -> IMU (FLU)
R_IC = np.array([
    [ 0.02514123, -0.08710579,  0.99588177],
    [-0.99910152,  0.03181017,  0.02800482],
    [-0.03411855, -0.99569106, -0.08622778]], dtype=float)

# IMU (FLU) -> Body (NED)
R_BI = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]], dtype=float)

R_BC = R_BI @ R_IC

P_BC = R_BI @ P_IC
