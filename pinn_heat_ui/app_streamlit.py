import numpy as np
import streamlit as st
import torch

from pathlib import Path
from pinnheat.model import PINNNet
from pinnheat.inference import load_checkpoint, predict_u
from pinnheat.exact import u_exact_heat

st.title("PINN Heat Equation UI")
st.write("Input x, t, alpha → outputs u_pred and u_exact")

@st.cache_resource
def load_model():
    device = "cpu"
    model = PINNNet(in_dim=3, width=64, depth=4)

    APP_DIR = Path(__file__).resolve().parent
    ckpt_path = APP_DIR / "checkpoints" / "model_best.pth"

    model, meta = load_checkpoint(str(ckpt_path), model, device=device)
    return model, meta, device

model, meta, device = load_model()

x = st.number_input("x", 0.0, 1.0, 0.5, 0.01)
t = st.number_input("t", 0.0, 1.0, 0.25, 0.01)
alpha = st.number_input("alpha", float(meta["alpha_min"]), float(meta["alpha_max"]), 0.175, 0.005)

if st.button("Predict"):
    u_pred = float(predict_u(model, x, t, alpha, meta, device=device))
    u_ex = float(u_exact_heat(x, t, alpha))
    st.success(f"u_pred = {u_pred:.8f}")
    st.info(f"u_exact = {u_ex:.8f}")
    st.write(f"abs error = {abs(u_ex-u_pred):.3e}")
import matplotlib.pyplot as plt

if st.button("Plot u(x)"):
    xs = np.linspace(0, 1, 200)
    ts = np.full_like(xs, t)
    alphas = np.full_like(xs, alpha)

    u_pred_line = [predict_u(model, xi, t, alpha, meta) for xi in xs]
    u_exact_line = u_exact_heat(xs, t, alpha)

    fig, ax = plt.subplots()
    ax.plot(xs, u_exact_line, label="Exact", linewidth=2)
    ax.plot(xs, u_pred_line, "--", label="PINN")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.legend()
    st.pyplot(fig)
