import numpy as np

def estimate_inertia_box(mass, L, W, H, add_rot_frac=0.2):
    Ixx = (1.0/12.0) * mass * (W**2 + H**2)
    Iyy = (1.0/12.0) * mass * (L**2 + H**2)
    Izz = (1.0/12.0) * mass * (L**2 + W**2)
    return (
        Ixx * (1 + add_rot_frac),
        Iyy * (1 + add_rot_frac),
        Izz * (1 + add_rot_frac),
    )

def build_M_matrix(mass, add_frac_x, add_frac_y, add_frac_z, L, W, H):
    mx = mass * (1 + add_frac_x)
    my = mass * (1 + add_frac_y)
    mz = mass * (1 + add_frac_z)
    Ixx, Iyy, Izz = estimate_inertia_box(mass, L, W, H)
    return np.diag([mx, my, mz, Ixx, Iyy, Izz])

def estimate_D_linear(rho, Cd, A, u0):
    return rho * Cd * A * u0

def build_D_matrix(u0):
    rho = 1000.0
    Cd_surge, A_surge = 0.8, 0.40 * 0.25
    Cd_sway,  A_sway  = 0.9, 0.46 * 0.25
    Cd_heave, A_heave = 1.0, 0.46 * 0.40

    Xu = estimate_D_linear(rho, Cd_surge, A_surge, u0)
    Yv = estimate_D_linear(rho, Cd_sway,  A_sway,  u0)
    Zw = estimate_D_linear(rho, Cd_heave, A_heave, u0)

    return np.diag([Xu, Yv, Zw, 0.5, 0.5, 0.8])
