import torch.nn as nn
from .base import BaseOperator


class Input(nn.Module, BaseOperator):
    def __init__(self):
        super(Input, self).__init__()

    def extra_repr(self):
        s = "scale={scale}"
        return s.format(**self.__dict__)

    def _get_name(self):
        return "QuantizedInput"

    @classmethod
    def from_float(cls, mod):
        activation_pre_process = mod
        scale = activation_pre_process.calculate_qparams()
        output_min_value = activation_pre_process.min_val
        output_max_value = activation_pre_process.max_val

        qinput = cls()
        qinput.scale = float(scale)
        qinput.qbit = 8
        qinput.output_min_value = output_min_value
        qinput.output_max_value = output_max_value

        return qinput

    def forward(self, x):
        output = x / self.scale
        output = self.clamp(output)

        return output
