import re
from loguru import logger
from onnx import helper

import torch.nn as nn

from .base_layer import BaseLayer


class PoolingLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(PoolingLayer, self).__init__(source_node, module, auto_gen)

    def get_pooling_attr(self):
        attr_dict = {
            "kernel_shape": [],
            "strides": [1, 1],
            "pads": [0, 0, 0, 0],
            "ceil_mode": False,
        }

        pool_dim = int(re.findall(r"(?:Pool)([0-9]d*?)", str(self._module))[0])

        if isinstance(self._module, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d)):
            dim = self._input_shape[0][2:]
            if isinstance(self._module.output_size, int):
                output_size = [self._module.output_size] * len(dim)
            else:
                output_size = self._module.output_size

            mod = [dim[i] % output_size[i] for i in range(0, len(dim))]
            if mod != [0] * len(mod):
                raise Exception(
                    "module %s Unsupported output size is not factor of input siz"
                    % (self._module)
                )

            k = [int(dim[i] / output_size[i]) for i in range(0, len(dim))]
            if len(k) == 1:
                attr_dict["strides"] = attr_dict["kernel_shape"] = [k[0]] * pool_dim
            else:
                attr_dict["strides"] = attr_dict["kernel_shape"] = k

            attr_dict["pads"] = [0] * (pool_dim * 2)

            return attr_dict
        kernel_size = self._module.kernel_size
        stride = self._module.stride
        padding = self._module.padding

        if isinstance(kernel_size, tuple):
            if len(kernel_size) == 1:
                attr_dict["kernel_shape"] = kernel_size * pool_dim
            else:
                attr_dict["kernel_shape"] = kernel_size
        else:
            attr_dict["kernel_shape"] = [kernel_size] * pool_dim

        if isinstance(stride, tuple):
            if len(stride) == 1:
                attr_dict["strides"] = stride * pool_dim
            else:
                attr_dict["strides"] = stride
        else:
            attr_dict["strides"] = [stride] * pool_dim

        if isinstance(padding, tuple):
            if len(padding) == 1:
                attr_dict["pads"] = padding * pool_dim * 2
            else:
                attr_dict["pads"] = padding * pool_dim
        else:
            attr_dict["pads"] = [padding] * pool_dim * 2

        attr_dict["ceil_mode"] = self._module.ceil_mode

        if isinstance(self._module, nn.AvgPool2d):
            attr_dict["pads"] = [0, 0, 0, 0]
        elif isinstance(self._module, nn.AvgPool1d):
            attr_dict["pads"] = [0, 0]

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):
        if name is not None:
            self._name = name

        if isinstance(self._module, (nn.MaxPool1d, nn.MaxPool2d)):
            attr_dict = self.get_pooling_attr()
            node = helper.make_node(
                "MaxPool", self._in_names, self._out_names, self._name, **attr_dict
            )
        elif isinstance(self._module, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d)):
            if isinstance(self._module.output_size, int):
                output_size = [self._module.output_size]
                output_size_len = 1
            else:
                output_size = [int(v) for v in self._module.output_size]
                output_size_len = len(self._module.output_size)
            if output_size == [1] * output_size_len:
                node = helper.make_node(
                    "GlobalAveragePool",
                    self._in_names,
                    self._out_names,
                    self._name,
                )
            else:
                attr_dict = self.get_pooling_attr()
                node = helper.make_node(
                    "AveragePool",
                    self._in_names,
                    self._out_names,
                    self._name,
                    **attr_dict
                )
        elif isinstance(self._module, (nn.AvgPool1d, nn.AvgPool2d)):
            attr_dict = self.get_pooling_attr()
            node = helper.make_node(
                "AveragePool", self._in_names, self._out_names, self._name, **attr_dict
            )
        logger.info("pooling_layer: " + self._name + " created")
        self._node.append(node)
