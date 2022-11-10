import warnings
from abc import ABCMeta, abstractmethod
from typing import Any, List, Tuple, Optional, Dict, Union
from functools import partial
import torch
import torch.nn as nn

ABC: Any = ABCMeta(str("ABC"), (object,), {})

AVIAIABLE_OBSERVERS = []


def register_observer(cls):

    AVIAIABLE_OBSERVERS.append(cls)

    return cls


def get_available_observers():
    return AVIAIABLE_OBSERVERS


def _with_args(cls_or_self, **kwargs):
    r"""Wrapper that allows creation of class factories.

    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances.

    Example::

        >>> Foo.with_args = classmethod(_with_args)
        >>> foo_builder = Foo.with_args(a=3, b=4).with_args(answer=42)
        >>> foo_instance1 = foo_builder()
        >>> foo_instance2 = foo_builder()
        >>> id(foo_instance1) == id(foo_instance2)
        False
    """

    class _PartialWrapper(object):
        def __init__(self, p):
            self.p = p

        def __call__(self, *args, **keywords):
            return self.p(*args, **keywords)

        def __repr__(self):
            return self.p.__repr__()

        with_args = _with_args

    r = _PartialWrapper(partial(cls_or_self, **kwargs))
    return r


class ObserverBase(ABC, nn.Module):
    r"""Base observer Module.
    Any observer implementation should derive from this class.

    Concrete observers should follow the same API. In forward, they will update
    the statistics of the observed Tensor. And they should provide a
    `calculate_qparams` function that computes the quantization parameters given
    the collected statistics.

    Args:
        dtype: Quantized data type
    """

    def __init__(self, dtype):
        super(ObserverBase, self).__init__()
        self.dtype = dtype

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def calculate_qparams(self, **kwargs):
        pass

    with_args = classmethod(_with_args)


class _ObserverBase(ObserverBase):
    _version = 2

    eps: torch.Tensor

    def __init__(
        self,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
    ):
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        super(_ObserverBase, self).__init__(dtype=dtype)
        self.qscheme = qscheme
        if reduce_range:
            warnings.warn(
                "Please use quant_min and quant_max to specify the range for observers. \
                    reduce_range will be deprecated in a future release of PyTorch."
            )
        self.reduce_range = reduce_range
        self.register_buffer(
            "eps", torch.tensor([torch.finfo(torch.float32).eps], **factory_kwargs)
        )
        assert self.qscheme in (
            torch.per_tensor_affine,
            torch.per_tensor_symmetric,
            torch.per_channel_affine,
            torch.per_channel_symmetric,
            torch.per_channel_affine_float_qparams,
        ), "Default Observer only works for per_tensor_affine, \
                per_tensor_symmetric, per_channel_affine, \
                per_channel_symmetric and per_channel_float_qparams quantization scheme"
        assert self.dtype in (
            torch.qint8,
            torch.quint8,
            torch.quint4x2,
        ), "Default Observer only works for qint8, quint8 and quint4x2 data type"
        self.has_customized_qrange = (quant_min is not None) and (quant_max is not None)
        if self.has_customized_qrange:
            self._validate_qmin_qmax(quant_min, quant_max)
        self.quant_min = quant_min
        self.quant_max = quant_max

    @torch.jit.export
    def _validate_qmin_qmax(self, quant_min: int, quant_max: int):
        assert (
            quant_min <= 0 <= quant_max
        ), "Used-specified quantization range must include 0."
        assert (
            quant_min < quant_max
        ), "qmin must be strictly less than qmax for user-specified quantization range."

    @torch.jit.export
    def _calculate_qmin_qmax(self):
        if self.dtype == torch.qint8:
            quant_min, quant_max = -128, 127
        elif self.dtype == torch.quint8:
            quant_min, quant_max = 0, 255

        return quant_min, quant_max

    @torch.jit.export
    def _calculate_qparams(self, min_val: torch.Tensor, max_val: torch.Tensor):
        r"""Calculates the quantization parameters, given min and max
        value tensors. Works for both per tensor and per channel cases

        Args:
            min_val: Minimum values per channel
            max_val: Maximum values per channel

        Returns:
            scales: Scales tensor of shape (#channels,)
        """
        if min_val.numel() == 0 or max_val.numel() == 0:
            warnings.warn(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
            return torch.tensor([1.0]), torch.tensor([0])

        if min_val.dim() == 0 or max_val.dim() == 0:
            if min_val == float("inf") and max_val == float("-inf"):
                warnings.warn(
                    "must run observer before calling calculate_qparams.\
                                        Returning default scale and zero point "
                )
                return torch.tensor([1.0]), torch.tensor([0])

            assert min_val <= max_val, "min {} should be less than max {}".format(
                min_val, max_val
            )
        else:
            assert torch.all(
                min_val <= max_val
            ), "min {} should be less than max {}".format(min_val, max_val)

        quant_min, quant_max = self._calculate_qmin_qmax()
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        scale = torch.ones(min_val_neg.size(), dtype=torch.float32)

        if (
            self.qscheme == torch.per_tensor_symmetric
            or self.qscheme == torch.per_channel_symmetric
        ):
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(quant_max - quant_min) / 2)
            scale = torch.max(scale, self.eps)
        elif self.qscheme == torch.per_channel_affine_float_qparams:
            scale = (max_val - min_val) / float(quant_max - quant_min)
            scale = torch.where(scale > self.eps, scale, torch.ones_like(scale))
        else:
            scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps)

        if len(scale.shape) == 0:
            scale = torch.tensor([float(scale)], dtype=scale.dtype)

        return scale


@register_observer
class MinMaxObserver(_ObserverBase):
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
    ):
        super(MinMaxObserver, self).__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
        )
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer("min_val", torch.tensor(float("inf"), **factory_kwargs))
        self.register_buffer("max_val", torch.tensor(float("-inf"), **factory_kwargs))
        if (
            self.qscheme == torch.per_tensor_symmetric
            and self.reduce_range
            and self.dtype == torch.quint8
        ):
            raise NotImplementedError(
                "Cannot reduce range for symmetric \
                                       quantization for quint8"
            )

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch._aminmax(x)
        min_val = torch.min(min_val_cur, self.min_val)
        max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        r"""Calculates the quantization parameters."""
        return self._calculate_qparams(self.min_val, self.max_val)

    @torch.jit.export
    def extra_repr(self):
        return "min_val={}, max_val={}".format(self.min_val, self.max_val)


@register_observer
class PerChannelMinMaxObserver(_ObserverBase):
    min_vals: torch.Tensor
    max_vals: torch.Tensor

    def __init__(
        self,
        ch_axis=0,
        dtype=torch.quint8,
        qscheme=torch.per_channel_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
    ):
        super(PerChannelMinMaxObserver, self).__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
        )
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.ch_axis = ch_axis
        self.register_buffer("min_vals", torch.tensor([], **factory_kwargs))
        self.register_buffer("max_vals", torch.tensor([], **factory_kwargs))
        if (
            self.qscheme == torch.per_channel_symmetric
            and self.reduce_range
            and self.dtype == torch.quint8
        ):
            raise NotImplementedError(
                "Cannot reduce range for symmetric quantization for quint8"
            )

    def forward(self, x_orig):
        return self._forward(x_orig)

    def _forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        min_vals = self.min_vals
        max_vals = self.max_vals
        x_dim = x.size()

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        # Need to match dtype of min/max because the updates to buffers
        # are done in place and types need to match for comparisons
        y = y.to(self.min_vals.dtype)
        y = torch.flatten(y, start_dim=1)
        if min_vals.numel() == 0 or max_vals.numel() == 0:
            min_vals, max_vals = torch._aminmax(y, 1)
        else:
            min_vals_cur, max_vals_cur = torch._aminmax(y, 1)
            min_vals = torch.min(min_vals_cur, min_vals)
            max_vals = torch.max(max_vals_cur, max_vals)
        self.min_vals.resize_(min_vals.shape)
        self.max_vals.resize_(max_vals.shape)
        self.min_vals.copy_(min_vals)
        self.max_vals.copy_(max_vals)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        return self._calculate_qparams(self.min_vals, self.max_vals)

    def extra_repr(self):
        return "min_val={}, max_val={}".format(self.min_vals, self.max_vals)


default_observer = MinMaxObserver.with_args()
default_weight_observer = PerChannelMinMaxObserver.with_args(
    dtype=torch.qint8, qscheme=torch.per_channel_symmetric
)
