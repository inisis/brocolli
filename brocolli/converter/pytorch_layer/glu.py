import torch
import torch.nn as nn


class GLU(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, x):
        out = torch.chunk(x, 2, self.dim)
        y = out[0] * torch.sigmoid(out[1])
        return y
