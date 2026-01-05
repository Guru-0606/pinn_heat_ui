import numpy as np
import torch

from .scaling import scale_to_minus1_plus1, scale_alpha

def load_checkpoint(path, model, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    meta = ckpt["meta"]
    return model, meta

@torch.no_grad()
def predict_u(model, x, t, alpha, meta, device="cpu"):
    x = np.asarray(x, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)

    x_min, x_max = meta["x_min"], meta["x_max"]
    t_min, t_max = meta["t_min"], meta["t_max"]
    a_min, a_max = meta["alpha_min"], meta["alpha_max"]

    x_t = torch.tensor(x).reshape(-1,1)
    t_t = torch.tensor(t).reshape(-1,1)
    a_t = torch.tensor(alpha).reshape(-1,1)

    x_s = scale_to_minus1_plus1(x_t, x_min, x_max)
    t_s = scale_to_minus1_plus1(t_t, t_min, t_max)
    a_s = scale_alpha(a_t, a_min, a_max)

    X = torch.cat([x_s, t_s, a_s], dim=1).to(device)
    u = model(X).cpu().numpy().reshape(x.shape)
    return u
