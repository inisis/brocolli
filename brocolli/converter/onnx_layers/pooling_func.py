import re
from loguru import logger
from onnx import helper

import torch.nn as nn

from .base_layer import BaseLayer


class PoolingFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(PoolingFunc, self).__init__(source_node, module, auto_gen)

    def get_pooling_attr(self, function_name):

        pool_dim = int(re.findall(r"(?:pool)([0-9]d*?)", str(function_name))[0])

        attr_dict = {
            "kernel_shape": [],
            "strides": [1, 1],
            "pads": [0, 0, 0, 0],
            "ceil_mode": False,
        }

        if (
            function_name == "adaptive_avg_pool1d"
            or function_name == "adaptive_avg_pool2d"
        ):
            output_size = self._source_node.args[1]
            dim = self._input_shape[0][2:]
            if isinstance(output_size, int):
                output_size = [output_size] * len(dim)
            else:
                output_size = output_size

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

        kernel_size = self._source_node.args[1]

        stride = self.get_value_by_key_or_index("stride", 2, kernel_size)
        padding = self.get_value_by_key_or_index("padding", 3, 0)
        ceil_mode = self.get_value_by_key_or_index("ceil_mode", 4, False)

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

        attr_dict["ceil_mode"] = ceil_mode

        if function_name == "avg_pool2d":
            attr_dict["pads"] = [0, 0, 0, 0]
        elif function_name == "avg_pool1d":
            attr_dict["pads"] = [0, 0]

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):
        if name is not None:
            self._name = name

        function_name = re.findall(
            r"(?:function|method) ([a-z|_|0-9]+.*?)", str(self._source_node.target)
        )[0]
        if function_name == "boolean_dispatch" and (
            "max_pool2d" in self._source_node.name
            or "max_pool1d" in self._source_node.name
        ):
            attr_dict = self.get_pooling_attr(self._source_node.name)
            node = helper.make_node(
                "MaxPool", self._in_names, self._out_names, self._name, **attr_dict
            )
        elif (
            function_name == "adaptive_avg_pool2d"
            or function_name == "adaptive_avg_pool1d"
        ):
            if isinstance(self._source_node.args[1], int):
                output_size = [1]
                output_size_len = 1
            else:
                output_size = [int(v) for v in self._source_node.args[1]]
                output_size_len = len(self._source_node.args[1])
            if output_size == [1] * output_size_len:
                node = helper.make_node(
                    "GlobalAveragePool",
                    self._in_names,
                    self._out_names,
                    self._name,
                )
            else:
                attr_dict = self.get_pooling_attr(function_name)
                node = helper.make_node(
                    "AveragePool",
                    self._in_names,
                    self._out_names,
                    self._name,
                    **attr_dict
                )
        elif function_name == "avg_pool2d" or function_name == "avg_pool1d":
            attr_dict = self.get_pooling_attr(function_name)
            node = helper.make_node(
                "AveragePool", self._in_names, self._out_names, self._name, **attr_dict
            )
        logger.info("pooling_layer: " + self._name + " created")
        self._node.append(node)
