import torch
import torch.nn as nn

class PINNNet(nn.Module):
    def __init__(self, in_dim=3, width=64, depth=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.Tanh())
        for _ in range(depth-1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
