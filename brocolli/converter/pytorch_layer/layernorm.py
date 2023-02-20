import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from .utils import transform_weight


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.dim = list(range(-len(self.normalized_shape), 0))
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(self.normalized_shape))
            self.bias = Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    @classmethod
    def from_torch(cls, mod):
        layernorm = cls(mod.normalized_shape, mod.eps, mod.elementwise_affine)
        if mod.elementwise_affine:
            state_dict = transform_weight(mod)
            layernorm.load_state_dict(state_dict)
        return layernorm

    def forward(self, x):
        mean = x.mean(dim=self.dim, keepdim=True)
        x -= mean
        var = (x**2).mean(dim=self.dim, keepdim=True)
        std = (var + self.eps).sqrt()
        y = x / std
        if self.elementwise_affine:
            y *= self.weight
            y += self.bias

        return y
