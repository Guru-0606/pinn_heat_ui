import numpy as np

def u_exact_heat(x, t, alpha):
    return np.sin(np.pi * x) * np.exp(-alpha * (np.pi**2) * t)
