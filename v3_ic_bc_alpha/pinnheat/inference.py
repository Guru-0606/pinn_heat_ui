import torch
from .model import PINNNet

def load_model(checkpoint_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)

    # --- support BOTH formats ---
    # (A) full checkpoint dict: {"model_state_dict": ...}
    # (B) raw state_dict: OrderedDict(...)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # infer input dimension from first layer weight
    # net.0.weight has shape [width, in_dim]
    in_dim = state_dict["net.0.weight"].shape[1]

    model = PINNNet(in_dim=in_dim, width=64, depth=4).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, device
