import re
from loguru import logger

import numpy as np
from onnx import helper
from onnx import TensorProto as tp

import torch.nn as nn

from .base_layer import BaseLayer


class ConvLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(ConvLayer, self).__init__(source_node, module, auto_gen)

    def get_conv_attr(self):
        conv_dim = int(re.findall(r"(?:Conv)([0-9]d*?)", str(self._module))[0])

        if isinstance(self._module, nn.Conv1d):  # con1d
            attr_dict = {
                "dilations": [1],  # list of ints defaults is 1
                "group": 1,  # int default is 1
                "kernel_shape": 1,  # list of ints If not present, should be inferred from input W.
                "pads": [0, 0],  # list of ints defaults to 0
                "strides": [1],  # list of ints  defaults is 1
            }

        else:
            attr_dict = {
                "dilations": [1, 1],  # list of ints defaults is 1
                "group": 1,  # int default is 1
                "kernel_shape": 1,  # list of ints If not present, should be inferred from input W.
                "pads": [0, 0, 0, 0],  # list of ints defaults to 0
                "strides": [1, 1],  # list of ints  defaults is 1
            }

        kernel_size = self._module.kernel_size
        stride = self._module.stride
        padding = self._module.padding
        dilation = self._module.dilation
        groups = self._module.groups

        if isinstance(dilation, tuple):
            attr_dict["dilations"] = dilation
        else:
            attr_dict["dilations"] = [dilation]

        if isinstance(kernel_size, tuple):
            attr_dict["kernel_shape"] = kernel_size
        else:
            attr_dict["kernel_shape"] = [kernel_size]

        if isinstance(stride, tuple):
            attr_dict["strides"] = stride
        else:
            attr_dict["strides"] = [stride]

        if isinstance(padding, tuple):
            if len(padding) == 1:
                attr_dict["pads"] = padding * conv_dim * 2
            else:
                attr_dict["pads"] = padding * 2
        else:
            attr_dict["pads"] = [padding] * conv_dim * 2

        attr_dict["group"] = groups

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):
        self.create_params(self._name + "_weight", self._module.weight.detach().numpy())
        if self._module.bias is not None:
            self.create_params(self._name + "_bias", self._module.bias.detach().numpy())

        attr_dict = self.get_conv_attr()
        logger.debug(attr_dict)
        node = helper.make_node(
            "Conv", self._in_names, self._out_names, self._name, **attr_dict
        )
        logger.info("conv_layer: " + self._name + " created")
        self._node.append(node)
