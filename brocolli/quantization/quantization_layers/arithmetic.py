import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseOperator
from .utils import _gen_lut
from .registry import register_quant_op


@register_quant_op("add")
class Add(nn.Module, BaseOperator):
    def __init__(self):
        super(Add, self).__init__()

    def extra_repr(self):
        s = "scale1={act_scale1}, scale2={act_scale2}, output_scale={output_scale}"
        return s.format(**self.__dict__)

    def _get_name(self):
        return "QuantizedReLU"

    @classmethod
    def from_float(cls, mod):
        assert hasattr(
            mod.activation_pre_process1, "qconfig"
        ), "float module must have qconfig defined."
        assert hasattr(
            mod.activation_pre_process2, "qconfig"
        ), "float module must have qconfig defined."
        activation_pre_process1 = mod.activation_pre_process1
        activation_pre_process2 = mod.activation_pre_process2
        activation_post_process = mod.activation_post_process
        act_scale1 = activation_pre_process1.calculate_qparams()
        act_scale2 = activation_pre_process2.calculate_qparams()
        output_scale = activation_post_process.calculate_qparams()

        qadd = cls()
        qadd.qbit = mod.activation_post_process.qbit
        qadd.act_scale1 = float(act_scale1)
        qadd.act_scale2 = float(act_scale2)
        qadd.output_scale = float(output_scale)
        qadd.output_min_value = activation_post_process.min_val
        qadd.output_max_value = activation_post_process.max_val

        return qadd

    def forward(self, input1, input2):
        input1 *= self.act_scale1
        input2 *= self.act_scale2
        out = torch.add(input1, input2)
        out = out / self.output_scale

        out = self.clamp(out)

        return out
