# PINN Heat Equation (Parametric)
This repository presents a Physics-Informed Neural Network (PINN) framework for solving the one-dimensional transient heat equation with a parametric thermal diffusivity. The trained model enables fast and accurate prediction of the temperature field for unseen space–time–parameter combinations and is deployed through an interactive Streamlit-based user interface.


α limits [0.05,0.30]
## PDE
∂u/∂t = α ∂²u/∂x²

## Initial & Boundary Conditions
u(x,0) = sin(πx)  
u(0,t) = u(1,t) = 0  

## Analytical Solution
u(x,t;α) = sin(πx) exp(-α π² t)

## Features
- Parametric PINN (x, t, α)
- Adam + L-BFGS training
- Residual-based adaptive refinement (RAR)
- Streamlit UI for interactive inference

## Run Locally
```bash
pip install -r requirements.txt
cd pinn_heat_ui
python -m streamlit run app_streamlit.py

## web link
https://pinnheatui-abembewxriq9cpngmdp8lu.streamlit.app/
