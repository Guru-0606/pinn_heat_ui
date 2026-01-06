# v3_ic_bc_alpha/pinnheat/scaling.py
import torch

def scale_to_minus1_plus1(x, x_min, x_max):
    return 2.0 * (x - x_min) / (x_max - x_min) - 1.0

def unscale_from_minus1_plus1(x_scaled, x_min, x_max):
    return 0.5 * (x_scaled + 1.0) * (x_max - x_min) + x_min

# These MUST match your training ranges
# (edit these numbers if your notebook used different bounds)
DEFAULT_RANGES = {
    "alpha": (0.05, 0.30),
    "A":     (0.50, 3.00),
    "k":     (1.00, 5.00),
    "B0":    (-0.50, 0.50),
    "B1":    (-0.50, 0.50),
}

def scale_param(p, p_min, p_max):
    return scale_to_minus1_plus1(p, p_min, p_max)

def unscale_param(p_s, p_min, p_max):
    return unscale_from_minus1_plus1(p_s, p_min, p_max)

def scale_alpha(alpha, alpha_min=None, alpha_max=None):
    if alpha_min is None or alpha_max is None:
        alpha_min, alpha_max = DEFAULT_RANGES["alpha"]
    return scale_param(alpha, alpha_min, alpha_max)

def scale_A(A, A_min=None, A_max=None):
    if A_min is None or A_max is None:
        A_min, A_max = DEFAULT_RANGES["A"]
    return scale_param(A, A_min, A_max)

def scale_k(k, k_min=None, k_max=None):
    if k_min is None or k_max is None:
        k_min, k_max = DEFAULT_RANGES["k"]
    return scale_param(k, k_min, k_max)

def scale_B0(B0, B0_min=None, B0_max=None):
    if B0_min is None or B0_max is None:
        B0_min, B0_max = DEFAULT_RANGES["B0"]
    return scale_param(B0, B0_min, B0_max)

def scale_B1(B1, B1_min=None, B1_max=None):
    if B1_min is None or B1_max is None:
        B1_min, B1_max = DEFAULT_RANGES["B1"]
    return scale_param(B1, B1_min, B1_max)
