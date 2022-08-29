import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseOperator
from .utils import _quantize_weight, _quantize_bias


class Linear(nn.Module, BaseOperator):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    def extra_repr(self):
        s = "in_features={in_features}, out_features={out_features}"
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)

    def _get_name(self):
        return "QuantizedLinear"

    @classmethod
    def from_float(cls, mod):
        weight_post_process = mod.qconfig.weight()
        weight_post_process(mod.weight)
        qweight, wt_scale = _quantize_weight(mod.weight.float(), weight_post_process)
        act_scale = mod.activation_pre_process.calculate_qparams()
        qbias = (
            _quantize_bias(mod.bias.float(), wt_scale * act_scale)
            if mod.bias is not None
            else None
        )
        output_scale = mod.activation_post_process.calculate_qparams()
        qlinear = cls(mod.in_features, mod.out_features, mod.bias)

        qlinear.qbit = mod.qbit
        qlinear.weight = torch.nn.Parameter(qweight, requires_grad=False)
        qlinear.bias = torch.nn.Parameter(qbias, requires_grad=False)
        qlinear.act_scale = float(act_scale)
        qlinear.wt_scale = wt_scale.reshape(1, -1)
        qlinear.output_scale = float(output_scale)
        qlinear.output_min_value = mod.activation_post_process.min_val
        qlinear.output_max_value = mod.activation_post_process.max_val

        return qlinear

    def forward(self, input):
        out = F.linear(
            input.to(torch.int64),
            self.weight.to(torch.int64),
            self.bias.to(torch.int64),
        )

        out = out * self.act_scale * self.wt_scale / self.output_scale
        out = self.clamp(out)

        return out
