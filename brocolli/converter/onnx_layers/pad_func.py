import re
from loguru import logger
import numpy as np
from onnx import helper
from onnx import TensorProto as tp

import torch.nn as nn

from .base_layer import BaseLayer


class PadFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(PadFunc, self).__init__(source_node, module, auto_gen)

    def gen_pad_attr(self):
        attr_dict = {"mode": "constant"}

        mode = self._source_node.kwargs["mode"]

        if mode == "replicate":
            attr_dict["mode"] = "edge"
        elif mode == "reflection":
            attr_dict["mode"] = "reflect"

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):
        if name is not None:
            self._name = name

        if params is None:
            output_dim = len(self._output_shape[0])
            pads = [0] * output_dim * 2
            padding = self._source_node.args[1]

            for idx in range(len(padding) // 2):
                pads[output_dim - idx - 1] = padding[idx * 2]
                pads[output_dim * 2 - idx - 1] = padding[idx * 2 + 1]

            params = np.array(pads)

            value = self._source_node.kwargs["value"]

            params = [np.array(pads), np.array(value)]

        if attr_dict is None:
            attr_dict = self.gen_pad_attr()

        self.create_params(self._name + "_pad", params[0])
        self.create_params(self._name + "_value", params[1])

        node = helper.make_node(
            "Pad", self._in_names, self._out_names, self._name, **attr_dict
        )

        logger.info("pad_layer: " + self._name + " created")
        self._node.append(node)
