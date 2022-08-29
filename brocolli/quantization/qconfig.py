import torch
import torch.nn as nn
from collections import namedtuple

from .observer import MinMaxObserver, PerChannelMinMaxObserver


class QConfig(namedtuple("QConfig", ["activation", "weight", "output"])):
    def __new__(cls, activation, weight, output):
        if isinstance(activation, nn.Module) or isinstance(weight, nn.Module):
            raise ValueError(
                "QConfig received observer instance, please pass observer class instead. "
                + "Use MyObserver.with_args(x=1) to override arguments to constructor if needed"
            )
        return super(QConfig, cls).__new__(cls, activation, weight, output)


def get_qconfig(bit, input_dtype=torch.qint8, output_dtype=torch.qint8):
    if bit == 8:
        return QConfig(
            activation=MinMaxObserver.with_args(dtype=input_dtype),
            weight=PerChannelMinMaxObserver.with_args(
                dtype=torch.qint8, qscheme=torch.per_channel_symmetric
            ),
            output=MinMaxObserver.with_args(dtype=output_dtype),
        )
    elif bit == 16:
        return QConfig(
            activation=MinMaxObserver.with_args(dtype=input_dtype),
            weight=PerChannelMinMaxObserver.with_args(
                dtype=torch.qint8, qscheme=torch.per_channel_symmetric
            ),
            output=MinMaxObserver.with_args(dtype=output_dtype),
        )
    else:
        raise ValueError("Quantization bit {} is not supported".format(bit))
