# v3_ic_bc_alpha/pinnheat/exact.py
import numpy as np

def u_exact_heat(x, t, alpha, A, k, B0, B1):
    """
    Exact solution used for your v3 demo:
      u(x,t) = g(x) + A*sin(k*pi*x)*exp(-alpha*(k*pi)^2*t)
      g(x) = B0 + (B1-B0)*x
    """
    g = B0 + (B1 - B0) * x
    return g + A * np.sin(k * np.pi * x) * np.exp(-alpha * (k * np.pi) ** 2 * t)
