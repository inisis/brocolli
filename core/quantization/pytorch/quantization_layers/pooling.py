import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseOperator
from .utils import _pair


class MaxPool2d(nn.Module, BaseOperator):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super(MaxPool2d, self).__init__()
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
        input = input.to(torch.float32)
        out = F.max_pool2d(
            input, self.kernel_size, self.stride, self.padding, self.dilation
        )
        out = out.to(torch.int64)

        return out
