import numpy as np

# --- Camera Matrix (Intrinsics)
FX = 634.390
FY = 636.006
CX = 654.090
CY = 369.740


# --- Distortion Coefficients (k1, k2, p1, p2, k3) ---
DIST_COEFFS = [-0.053665, 0.0314185, -0.00216851, 0.00262024, 0.0]


# --- ASEPP Physical Offsets (Camera relative to CoG) ---
P_CB_X = -0.17
P_CB_Y = 0.0
P_CB_Z = 0.02


# --- IBVS Control Parameters ---
LAMBDA_P = np.array([0.8, 0.8, 0.6, 0.0, 0.5, 0.0]) # Surge - Sway - Heave - Roll - Pitch - Yaw
# LAMBDA_P = 0.5
mu = 0.2
Z_DES = 1.5
TAG_SIZE = 0.2065
PATCH = 2

W_diag = np.array([1.0, 1.0, 0.5])
W = np.diag(np.tile(W_diag, 4))
# W = np.kron(np.eye(4), W_diag)



# --- Velocity Limits ---
MAX_LIN_VEL = 5
MAX_ANG_VEL = 4


# --- QGC Port Configuration ---
QGC_IP = "192.168.4.1"
QGC_PORT = 5600


# --- Transformation Matrix ---
# Camera X right | Y down  | Z front
# U-ROVs X front | Y right | Z down
R_CB = np.array([[0,0,1],
                 [1,0,0],
                 [0,1,0]], dtype=float) # CAM to NED

P_CB = np.array([P_CB_X, P_CB_Y, P_CB_Z])
