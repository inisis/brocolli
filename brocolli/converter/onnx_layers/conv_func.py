import re
from loguru import logger

import numpy as np
from onnx import helper
from onnx import TensorProto as tp

import torch.nn as nn

from .base_layer import BaseLayer


class ConvFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(ConvFunc, self).__init__(source_node, module, auto_gen)

    def get_conv_attr(self):
        function_name = re.findall(
            r"(?:function|method) ([a-z|_|0-9]+.*?)", str(self._source_node.target)
        )[0]
        conv_dim = int(re.findall(r"(?:conv)([0-9]d*?)", str(function_name))[0])

        if isinstance(self._module, nn.Conv1d):  # con1d
            attr_dict = {
                "dilations": [1],  # list of ints defaults is 1
                "group": 1,  # int default is 1
                "pads": [0, 0],  # list of ints defaults to 0
                "strides": [1],  # list of ints  defaults is 1
            }

        else:
            attr_dict = {
                "dilations": [1, 1],  # list of ints defaults is 1
                "group": 1,  # int default is 1
                "pads": [0, 0, 0, 0],  # list of ints defaults to 0
                "strides": [1, 1],  # list of ints  defaults is 1
            }

        stride = self._source_node.args[3]
        padding = self._source_node.args[4]
        dilation = self._source_node.args[5]
        groups = self._source_node.args[6]

        if isinstance(dilation, tuple):
            attr_dict["dilations"] = dilation
        else:
            attr_dict["dilations"] = [dilation]

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
        attr_dict = self.get_conv_attr()
        logger.debug(attr_dict)
        node = helper.make_node(
            "Conv", self._in_names, self._out_names, self._name, **attr_dict
        )
        logger.info("conv_layer: " + self._name + " created")
        self._node.append(node)
