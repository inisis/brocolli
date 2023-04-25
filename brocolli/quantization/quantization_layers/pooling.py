import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseOperator
from .utils import _pair


class MaxPool(nn.Module, BaseOperator):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super(MaxPool, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

    def extra_repr(self):
        s = "kernel_size={kernel_size}, stride={stride}, padding={padding}, dilation={dilation}"
        return s.format(**self.__dict__)

    def _get_name(self):
        return "QuantizedMaxPool2d"

    @classmethod
    def from_float(cls, mod):
        qmaxpool = cls(mod.kernel_size, mod.stride, mod.padding, mod.dilation)

        return qmaxpool

    def forward(self, input):
        input = input.to(torch.float64)
        out = F.max_pool2d(
            input, self.kernel_size, self.stride, self.padding, self.dilation
        )
        out = out.to(torch.float64)

        return out


class AdaptiveAvgPool(nn.Module, BaseOperator):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super(AdaptiveAvgPool, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

    def extra_repr(self):
        s = "kernel_size={kernel_size}, stride={stride}, padding={padding}, dilation={dilation}"
        return s.format(**self.__dict__)

    def _get_name(self):
        return "QuantizedAdaptiveAvgPool2d"

    @classmethod
    def from_float(cls, mod):
        if isinstance(mod.output_size, int):
            output_size = (mod.output_size, mod.output_size)
        else:
            output_size = mod.output_size
        kernel_size = stride = (
            mod.activation_pre_process.input_shape[2] // output_size[0],
            mod.activation_pre_process.input_shape[3] // output_size[1],
        )
        qmaxpool = cls(kernel_size, stride, 0, 0)

        return qmaxpool

    def forward(self, input):
        input = input.to(torch.float64)
        out = F.avg_pool2d(input, self.kernel_size, self.stride, self.padding)
        out = out.to(torch.int64)

        return out
