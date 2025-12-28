import numpy as np

# ibvs_control/constants.py
# --- Camera Matrix (Intrinsics) from RGB Column ---
FX = 634.390
FY = 636.006
CX = 654.090
CY = 369.740

# --- Distortion Coefficients (k1, k2, p1, p2, k3) ---
# Note: k3 is not in your table, so we use 0.0
DIST_COEFFS = [-0.053665, 0.0314185, -0.00216851, 0.00262024, 0.0]

# --- BlueROV2 Physical Offsets (Camera relative to CoG) ---
P_CB_X = -0.17
P_CB_Y = 0.0
P_CB_Z = 0.02

# --- IBVS Control Parameters ---
LAMBDA_P = 0.0001
K_ACNHOR = 0.3
Z_DES = 1.0
TAG_SIZE = 0.207
PATCH = 2

# --- Velocity Limits ---
MAX_LIN_VEL = 0.3
MAX_ANG_VEL = 0.4

# --- QGC Port Configuration ---
QGC_IP = "192.168.4.1"
QGC_PORT = 5600

# --- Transformation Matrix ---
R_CB = np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=float)
P_CB = np.array([P_CB_X, P_CB_Y, P_CB_Z])

MAX_OFFSET = 0.1

MAX_TAU_Z = 0.3
MAX_TAU_YAW = 0.3
MAX_TAU_X = 0.3
MAX_TAU_Y = 0.3
