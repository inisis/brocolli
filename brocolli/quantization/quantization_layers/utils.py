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


def _quantize_weight(float_wt, observer, bit_width=8):
    wt_scale = observer.calculate_qparams()
    wt_scale_tmp = torch.tensor(wt_scale)
    wt_scale_tmp = wt_scale_tmp * 2 ** (bit_width - 1) / 2 ** (bit_width - 1)
    extra_dims = (1,) * (float_wt.dim() - 1)
    wt_scale_tmp = wt_scale_tmp.view(-1, *extra_dims)
    wt_scale_tmp = torch.where(
        wt_scale_tmp == 0, torch.tensor(1.0, dtype=torch.float32), wt_scale_tmp
    )
    q_float_wt = torch.div(float_wt, wt_scale_tmp)
    q_float_wt = torch.where(
        wt_scale_tmp == 0, torch.tensor(0.0, dtype=torch.float32), q_float_wt
    )
    q_float_wt = torch.round(q_float_wt)
    min_val = -(2 ** (bit_width - 1))
    max_val = 2 ** (bit_width - 1) - 1
    out_tensor = torch.clamp(q_float_wt, min_val, max_val)
    qweight = out_tensor.to(torch.int64)

    return qweight, wt_scale


def _quantize_bias(float_bias, scale, bit_width=8):
    bias_scale = torch.tensor(scale)
    extra_dims = (1,) * (float_bias.dim() - 1)
    bias_scale = bias_scale.view(-1, *extra_dims)
    bias_scale_tmp = torch.where(
        bias_scale == 0, torch.tensor(1.0, dtype=torch.float32), bias_scale
    )
    q_float_bias = torch.div(float_bias, bias_scale_tmp)
    q_float_bias = torch.where(
        bias_scale == 0, torch.tensor(0.0, dtype=torch.float32), q_float_bias
    )
    q_float_bias = torch.round(q_float_bias)
    min_val = -(2 ** (bit_width - 1))
    max_val = 2 ** (bit_width - 1) - 1
    out_tensor = torch.clamp(q_float_bias, min_val, max_val)
    qbias = out_tensor.to(torch.int64)

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
