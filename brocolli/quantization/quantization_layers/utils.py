import torch
import collections
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def _quantize_weight(float_wt, observer):
    wt_scale = observer.calculate_qparams()
    if observer.qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]:
        qweight = torch.quantize_per_tensor(
            float_wt, float(wt_scale), torch.zeros_like(wt_scale), torch.qint8
        )
    elif observer.qscheme in [torch.per_channel_symmetric, torch.per_channel_affine]:
        wt_axis = observer.ch_axis
        qweight = torch.quantize_per_channel(
            float_wt,
            wt_scale.to(torch.double),
            torch.zeros_like(wt_scale).to(torch.int64),
            wt_axis,
            torch.qint8,
        ).int_repr()
    elif observer.qscheme in [torch.per_channel_affine_float_qparams]:
        qweight = torch.quantize_per_channel(
            float_wt, wt_scale.to(torch.float), 0, observer.ch_axis, observer.dtype
        )
    else:
        raise ValueError("Unexpected qscheme " + observer.qscheme)
    return qweight, wt_scale


def _quantize_bias(float_bias, scale):
    qbias = torch.quantize_per_channel(
        float_bias,
        scale.to(torch.double),
        torch.zeros_like(scale).to(torch.int64),
        0,
        torch.qint32,
    ).int_repr()

    return qbias


def _gen_lut(function, act_scale, output_scale, start, end):
    weight = []
    for idx in range(start, end):
        input = idx * act_scale
        output = function(input) / output_scale
        output = output.to(torch.int64)
        weight.append(output)
    weight = torch.stack(weight)

    return weight
