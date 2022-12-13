import re
from loguru import logger
import numpy as np
from onnx import helper
from onnx import TensorProto as tp

import torch.nn as nn

from .base_layer import BaseLayer


class PadLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(PadLayer, self).__init__(source_node, module, auto_gen)

    def gen_pad_attr(self):
        attr_dict = {"mode": "constant"}
        mode = re.findall(r"([a-z|A-Z]+.)(?:Pad[0-9]d*?)", str(self._module))[0]

        if mode == "Replication":
            attr_dict["mode"] = "edge"
        elif mode == "Reflection":
            attr_dict["mode"] = "reflect"

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):
        if name is not None:
            self._name = name

        if params is None:
            output_dim = len(self._output_shape[0])
            pads = [0] * output_dim * 2
            padding = self._module.padding

            for idx in range(len(padding) // 2):
                pads[output_dim - idx - 1] = padding[idx * 2]
                pads[output_dim * 2 - idx - 1] = padding[idx * 2 + 1]

            if hasattr(self._module, "value"):
                value = self._module.value
            else:
                value = 0.0
            params = [np.array(pads), np.array(value)]

        self.create_params(self._name + "_pad", params[0])
        self.create_params(self._name + "_value", params[1])

        if attr_dict is None:
            attr_dict = self.gen_pad_attr()

        node = helper.make_node(
            "Pad", self._in_names, self._out_names, self._name, **attr_dict
        )

        logger.info("pad_layer: " + self._name + " created")
        self._node.append(node)
