import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Page config + header
# -----------------------------
st.set_page_config(page_title="PINN Heat UI v3", layout="wide")
st.title("🔥 PINN Heat Equation UI — v3 (IC + BC + α)")
st.caption("Repo module: `v3_ic_bc_alpha/app_streamlit.py`")

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Controls")

    st.subheader("Domain point")
    x = st.slider("x", 0.0, 1.0, 0.50, 0.01)
    t = st.slider("t", 0.0, 1.0, 0.25, 0.01)

    st.divider()
    st.subheader("PDE parameter")
    alpha = st.slider("alpha (diffusivity)", 0.05, 0.30, 0.175, 0.005)

    st.divider()
    st.subheader("Initial condition (IC)")
    A = st.slider("A (amplitude)", 0.5, 3.0, 0.5, 0.1)
    k = st.slider("k (mode / wavenumber)", 1, 5, 1, 1)

    st.divider()
    st.subheader("Boundary condition (BC)")
    B0 = st.slider("B0  (x=0)", -1.0, 1.0, 0.0, 0.05)
    B1 = st.slider("B1  (x=1)", -1.0, 1.0, 0.0, 0.05)

    st.divider()
    st.subheader("Quick presets")
    preset = st.selectbox(
        "Select preset",
        [
            "Custom",
            "Baseline (A=0.5,k=1,B0=0,B1=0)",
            "High IC (A=3,k=5,B0=0,B1=0)",
            "Shifted BC (A=0.5,k=1,B0=0.5,B1=0)",
            "Mixed BC (A=0.5,k=1,B0=-0.5,B1=0.5)",
        ],
        index=1
    )

    if preset != "Custom":
        if preset == "Baseline (A=0.5,k=1,B0=0,B1=0)":
            A, k, B0, B1 = 0.5, 1, 0.0, 0.0
        elif preset == "High IC (A=3,k=5,B0=0,B1=0)":
            A, k, B0, B1 = 3.0, 5, 0.0, 0.0
        elif preset == "Shifted BC (A=0.5,k=1,B0=0.5,B1=0)":
            A, k, B0, B1 = 0.5, 1, 0.5, 0.0
        elif preset == "Mixed BC (A=0.5,k=1,B0=-0.5,B1=0.5)":
            A, k, B0, B1 = 0.5, 1, -0.5, 0.5

# -----------------------------
# Summary card
# -----------------------------
c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])
c1.metric("x", f"{x:.3f}")
c2.metric("t", f"{t:.3f}")
c3.metric("α", f"{alpha:.3f}")
c4.metric("IC/BC", f"A={A:.2f}, k={k}, B0={B0:.2f}, B1={B1:.2f}")

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📌 Point prediction", "🗺️ Field plot", "🧪 Diagnostics"])

with tab1:
    st.subheader("Predict at a single point (x,t)")

    colA, colB = st.columns([1, 2])
    with colA:
        do_predict = st.button("Predict", use_container_width=True)
    with colB:
        st.info("Uses current x,t,α,A,k,B0,B1. Make sure your v3 model expects these inputs.")

    if do_predict:
        # TODO: replace with your real function
        # u_pred, u_exact = predict_point(model, x, t, alpha, A, k, B0, B1)
        u_pred, u_exact = np.nan, np.nan

        m1, m2, m3 = st.columns(3)
        m1.metric("u_pred", f"{u_pred:.6f}" if np.isfinite(u_pred) else "NA")
        m2.metric("u_exact", f"{u_exact:.6f}" if np.isfinite(u_exact) else "NA")
        if np.isfinite(u_pred) and np.isfinite(u_exact):
            m3.metric("|error|", f"{abs(u_pred-u_exact):.3e}")
        else:
            m3.metric("|error|", "NA")

with tab2:
    st.subheader("Field plot u(x,t) heatmaps")
    st.caption("This is the 1×4 layout: Exact | PINN | Abs Error | Time-slices")

    Nx = st.slider("Nx (spatial resolution)", 51, 401, 201, 50)
    Nt = st.slider("Nt (time resolution)", 51, 401, 201, 50)
    time_slices = st.multiselect(
        "Time slices",
        options=[0.0, 0.25, 0.5, 0.75, 1.0],
        default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )

    if st.button("Plot field", use_container_width=True):
        # TODO: replace with your real function
        # xg, tg, U_exact, U_pred, AbsErr, rel_l2 = predict_on_grid(model, alpha, A, k, B0, B1, Nx=Nx, Nt=Nt)
        xg = np.linspace(0, 1, Nx)
        tg = np.linspace(0, 1, Nt)
        U_exact = np.zeros((Nt, Nx))
        U_pred  = np.zeros((Nt, Nx))
        AbsErr  = np.zeros((Nt, Nx))
        rel_l2 = np.nan

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))

        im0 = axes[0].imshow(U_exact, extent=[0, 1, 1, 0], aspect="auto")
        axes[0].set_title("Exact")
        axes[0].set_xlabel("x"); axes[0].set_ylabel("t")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(U_pred, extent=[0, 1, 1, 0], aspect="auto")
        axes[1].set_title("PINN")
        axes[1].set_xlabel("x"); axes[1].set_ylabel("t")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(AbsErr, extent=[0, 1, 1, 0], aspect="auto")
        axes[2].set_title("Absolute error")
        axes[2].set_xlabel("x"); axes[2].set_ylabel("t")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        axes[3].set_title("Time slices")
        axes[3].set_xlabel("x"); axes[3].set_ylabel("u(x,t)")
        for ts in time_slices:
            idx = int(round(ts * (Nt - 1)))
            axes[3].plot(xg, U_exact[idx, :], label=f"Exact t={ts:.2f}")
            axes[3].plot(xg, U_pred[idx, :], "--", label=f"PINN t={ts:.2f}")
        axes[3].legend(fontsize=7, ncol=1)

        st.pyplot(fig, clear_figure=True)
        st.write(f"**relL2:** {rel_l2}")

with tab3:
    st.subheader("Diagnostics")
    st.write("Use this area for:")
    st.markdown("""
    - PDE residual checks on random collocation points  
    - “unseen combination” tests  
    - model input-dimension sanity checks  
    - loss curves (Adam vs LBFGS)  
    """)

    if st.button("Show model info (placeholder)", use_container_width=True):
        st.code("TODO: print model input dim, first layer weight shape, checkpoint path, device")
