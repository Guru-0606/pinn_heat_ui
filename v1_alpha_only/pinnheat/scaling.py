import torch

def scale_to_minus1_plus1(val, vmin, vmax):
    return 2.0*(val - vmin)/(vmax - vmin) - 1.0

def scale_alpha(alpha, alpha_min, alpha_max):
    return scale_to_minus1_plus1(alpha, alpha_min, alpha_max)
