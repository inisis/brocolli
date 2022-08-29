import torch.nn as nn
from .base import BaseOperator


class Output(nn.Module, BaseOperator):
    def __init__(self):
        super(Output, self).__init__()

    def extra_repr(self):
        s = "scale={scale}"
        return s.format(**self.__dict__)

    def _get_name(self):
        return "DeQuantizedOutput"

    @classmethod
    def from_float(cls, mod):
        # assert hasattr(mod, "qconfig"), \
        #     "Output float module must have qconfig defined."
        # activation_post_process = mod.activation_post_process
        # scale = activation_post_process.calculate_qparams()

        qoutput = cls()
        qoutput.scale = float(mod.output_scale)

        return qoutput

    def forward(self, x):
        output = x * self.scale

        return output
