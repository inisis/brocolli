import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseOperator


class ReLU(nn.Module, BaseOperator):
    def __init__(self):
        super(ReLU, self).__init__()

    def extra_repr(self):
        s = "scale={act_scale}, output_scale={output_scale}"
        return s.format(**self.__dict__)

    def _get_name(self):
        return "QuantizedInput"

    @classmethod
    def from_float(cls, mod):
        assert hasattr(mod, "qconfig"), "Relu float module must have qconfig defined."
        activation_pre_process = mod.activation_pre_process
        activation_post_process = mod.activation_post_process
        act_scale = activation_pre_process.calculate_qparams()
        output_scale = activation_post_process.calculate_qparams()

        qrelu = cls()
        qrelu.qbit = mod.qbit
        qrelu.act_scale = float(act_scale)
        qrelu.output_scale = float(output_scale)
        qrelu.output_min_value = activation_post_process.min_val
        qrelu.output_max_value = activation_post_process.max_val

        return qrelu

    def forward(self, input):
        out = F.relu(input.to(torch.int64))

        out = out * self.act_scale / self.output_scale
        out = self.clamp(out)

        return out
