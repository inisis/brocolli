import torch
import torch.nn as nn
from collections import namedtuple

from .observer import MinMaxObserver, PerChannelMinMaxObserver, LSQObserver


class QConfig(namedtuple("QConfig", ["activation", "weight"])):
    def __new__(cls, activation, weight):
        if isinstance(activation, nn.Module) or isinstance(weight, nn.Module):
            raise ValueError(
                "QConfig received observer instance, please pass observer class instead. "
                + "Use MyObserver.with_args(x=1) to override arguments to constructor if needed"
            )
        return super(QConfig, cls).__new__(cls, activation, weight)


def get_qconfig(bit, input_dtype=torch.qint8, lsq=False):
    if lsq:
        return QConfig(
            activation=LSQObserver.with_args(
                dtype=input_dtype, all_positive=True, symmetric=False, per_channel=False
            ),
            weight=LSQObserver.with_args(
                dtype=input_dtype, all_positive=True, symmetric=False, per_channel=True
            ),
        )
    elif bit == 8:
        return QConfig(
            activation=MinMaxObserver.with_args(
                dtype=input_dtype, qscheme=torch.per_tensor_symmetric
            ),
            weight=PerChannelMinMaxObserver.with_args(
                dtype=torch.qint8, qscheme=torch.per_channel_symmetric
            ),
        )
    elif bit == 16:
        return QConfig(
            activation=MinMaxObserver.with_args(
                dtype=input_dtype, qscheme=torch.per_tensor_symmetric
            ),
            weight=PerChannelMinMaxObserver.with_args(
                dtype=torch.qint8, qscheme=torch.per_channel_symmetric
            ),
        )
    else:
        raise ValueError("Quantization bit {} is not supported".format(bit))
