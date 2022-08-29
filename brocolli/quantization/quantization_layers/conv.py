import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseOperator
from .utils import _pair, _quantize_weight, _quantize_bias


_SUPPORTED_PADDING = {"zeros", "reflect"}


class _ConvNd(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        dtype=None,
    ):
        raise NotImplementedError

    def _init(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode="zeros",
        dtype=None,
    ):
        factory_kwargs = {"dtype": dtype}
        super(_ConvNd, self).__init__()

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if padding_mode not in _SUPPORTED_PADDING:
            raise ValueError(
                "'padding_mode' {} is not supported by quantized convolution".format(
                    padding_mode
                )
            )
        self.padding_mode = padding_mode

        if self.transposed:
            weight_shape = [in_channels, out_channels // self.groups]
        else:
            weight_shape = [out_channels, in_channels // self.groups]
        qweight = torch._empty_affine_quantized(
            weight_shape + list(kernel_size),
            scale=1,
            zero_point=0,
            dtype=torch.qint8,
            **{k: v for k, v in factory_kwargs.items() if k != "dtype"}
        )
        bias_float = (
            torch.zeros(
                out_channels,
                dtype=torch.float,
                **{k: v for k, v in factory_kwargs.items() if k != "dtype"}
            )
            if bias
            else None
        )

        self.weight = qweight
        self.bias = bias_float
        self.scale = 1.0

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}, act_scale={act_scale}, output_scale={output_scale}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)

    def __deepcopy__(self, memo):
        new_instance = type(self).__new__(type(self))
        torch.nn.Module.__init__(new_instance)
        state = self.__getstate__()
        new_instance.__setstate__(state)
        return new_instance

    def __copy__(self):
        return self.__deepcopy__({})

    @classmethod
    def get_qconv(
        cls,
        mod,
        activation_pre_process,
        weight_post_process=None,
        activation_post_process=None,
    ):
        r"""Creates a qconv object and returns it."""
        if weight_post_process is None:
            weight_post_process = mod.qconfig.weight()

        weight_post_process(mod.weight)
        qweight, wt_scale = _quantize_weight(mod.weight.float(), weight_post_process)
        act_scale = activation_pre_process.calculate_qparams()
        qbias = (
            _quantize_bias(mod.bias.float(), wt_scale * act_scale)
            if mod.bias is not None
            else None
        )
        output_scale = activation_post_process.calculate_qparams()

        assert (
            weight_post_process.dtype == torch.qint8
        ), "Weight observer must have a dtype of qint8"

        qconv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,  # type: ignore[call-arg]
            mod.stride,
            mod.padding,
            mod.dilation,
            mod.groups,
            mod.bias is not None,
            mod.padding_mode,
        )

        qconv.qbit = mod.qbit
        qconv.weight = torch.nn.Parameter(qweight, requires_grad=False)
        qconv.bias = torch.nn.Parameter(qbias, requires_grad=False)
        qconv.act_scale = float(act_scale)
        qconv.wt_scale = wt_scale.reshape(1, -1, 1, 1)
        qconv.output_scale = float(output_scale)
        qconv.output_min_value = activation_post_process.min_val
        qconv.output_max_value = activation_post_process.max_val

        return qconv

    @staticmethod
    def from_float(cls, mod):
        assert type(mod) == cls._FLOAT_MODULE, (
            " brocolli."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
            + " but got:"
            + str(type(mod))
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined."
        activation_pre_process = mod.activation_pre_process
        weight_post_process = mod.qconfig.weight()
        activation_post_process = mod.activation_post_process
        return cls.get_qconv(
            mod, activation_pre_process, weight_post_process, activation_post_process
        )


class Conv2d(_ConvNd, BaseOperator):
    _FLOAT_MODULE = nn.Conv2d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        dtype=None,
    ):
        factory_kwargs = {"dtype": dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(Conv2d, self)._init(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs
        )

    def _get_name(self):
        return "QuantizedConv2d"

    def forward(self, input):
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")

        out = F.conv2d(
            input.to(torch.int64),
            self.weight.to(torch.int64),
            self.bias.to(torch.int64),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        out = out * self.act_scale * self.wt_scale / self.output_scale
        out = self.clamp(out)

        return out

    @classmethod
    def from_float(cls, mod):
        r"""Creates a quantized module from a float module or qparams_dict.

        Args:
            mod (Module): a float module, either produced by torch.quantization
              utilities or provided by the user
        """
        return _ConvNd.from_float(cls, mod)
