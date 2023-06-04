import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger

from .base import BaseOperator
from .utils import _pair
from .registry import register_quant_op


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
        input = input.to(torch.float32)
        out = F.max_pool2d(
            input, self.kernel_size, self.stride, self.padding, self.dilation
        )
        out = out.to(torch.float32)

        return out


@register_quant_op(torch.nn.AdaptiveAvgPool2d)
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
        act_scale = mod.activation_pre_process.calculate_qparams()
        logger.debug(
            f"{mod.name} activation scale: {act_scale}, max_val: {mod.activation_pre_process.max_val}, min_val: {mod.activation_pre_process.min_val}"
        )

        output_scale = mod.activation_post_process.calculate_qparams()
        logger.debug(
            f"{mod.name} output scale: {output_scale}, max_val: {mod.activation_post_process.max_val}, min_val: {mod.activation_post_process.min_val}"
        )

        if isinstance(mod.output_size, int):
            output_size = (mod.output_size, mod.output_size)
        else:
            output_size = mod.output_size
        kernel_size = stride = (
            mod.activation_pre_process.input_shape[2] // output_size[0],
            mod.activation_pre_process.input_shape[3] // output_size[1],
        )
        qmaxpool = cls(kernel_size, stride, 0, 0)
        qmaxpool.qbit = mod.activation_post_process.qbit
        qmaxpool.act_scale = act_scale
        qmaxpool.output_scale = output_scale
        qmaxpool.output_min_value = mod.activation_post_process.min_val
        qmaxpool.output_max_value = mod.activation_post_process.max_val

        return qmaxpool

    def forward(self, input):
        input *= self.act_scale
        out = F.avg_pool2d(input, self.kernel_size, self.stride, self.padding)

        out = out / self.output_scale
        out = self.clamp(out)

        return out
