import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseOperator
from .utils import _gen_lut


class ReLU(nn.Module, BaseOperator):
    def __init__(self):
        super(ReLU, self).__init__()

    def extra_repr(self):
        s = "scale={act_scale}, output_scale={output_scale}"
        return s.format(**self.__dict__)

    def _get_name(self):
        return "QuantizedReLU"

    @classmethod
    def from_float(cls, mod, lut=False):
        assert hasattr(mod, "qconfig"), "Relu float module must have qconfig defined."
        activation_pre_process = mod.activation_pre_process
        activation_post_process = mod.activation_post_process
        act_scale = activation_pre_process.calculate_qparams()
        output_scale = activation_post_process.calculate_qparams()

        if lut:
            if activation_pre_process.min_val >= 0:
                lut_start = 0
                zero_point = 0
                lut_end = 255
            else:
                lut_start = -128
                zero_point = 128
                lut_end = 127
            lut_weight = _gen_lut(F.relu, act_scale, output_scale, lut_start, lut_end)
            lut_weight = torch.nn.Parameter(lut_weight, requires_grad=False)
            qrelu = cls()
            qrelu.qbit = mod.qbit
            qrelu.lut_weight = lut_weight
            qrelu.zero_point = zero_point
            qrelu.act_scale = float(act_scale)
            qrelu.output_scale = float(output_scale)
            qrelu.output_min_value = activation_post_process.min_val
            qrelu.output_max_value = activation_post_process.max_val
        else:
            qrelu = cls()
            qrelu.qbit = mod.qbit
            qrelu.act_scale = float(act_scale)
            qrelu.output_scale = float(output_scale)
            qrelu.output_min_value = activation_post_process.min_val
            qrelu.output_max_value = activation_post_process.max_val

        return qrelu

    def forward(self, input):
        if hasattr(self, "lut_weight"):
            input = input.to(torch.int64) + self.zero_point
            out = F.embedding(input, self.lut_weight).squeeze(-1)
        else:
            out = F.relu(input.to(torch.int64))
            out = out * self.act_scale / self.output_scale

        out = self.clamp(out)

        return out
