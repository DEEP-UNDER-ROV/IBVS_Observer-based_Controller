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


# --- ASEPP Physical Offsets (Camera relative to CoG) ---
P_CB_X = 0.17
P_CB_Y = -0.0475
P_CB_Z = 0.0


# --- IBVS Control Parameters ---
# Kp = np.array([0.6, 0.6, 0.5, 0.0, 0.3, 0.0]) # Surge - Sway - Heave - Roll - Pitch - Yaw
# Kd = np.array([0.2, 0.2, 0.2, 0.0, 0.05, 0.0])
# Ki = np.array([0.08, 0.08, 0.08, 0.0, 0.05, 0.0])

mu = 0.2
Z_DES = 1.5
TAG_SIZE = 0.2065
PATCH = 2

# ---------------- Gains (TUNE IN WATER) ----------------
K_SURGE = 400
K_SWAY  = 400
K_HEAVE = 400
K_ROLL = 400
K_PITCH = 400
K_YAW   = 400
HEAVE_BIAS = 20


# --- Velocity Limits ---
MAX_LIN_VEL = 0.2
MAX_ANG_VEL = 0.08


# --- QGC Port Configuration ---
QGC_IP = "192.168.2.1"
QGC_PORT = 5600


# --- Transformation Matrix ---
# Camera X right | Y down  | Z front
# U-ROVs X front | Y right | Z down
R_CB = np.array([[0,0,1],
                 [1,0,0],
                 [0,1,0]], dtype=float) # CAM to NED

P_CB = np.array([P_CB_X, P_CB_Y, P_CB_Z])
