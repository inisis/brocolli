from loguru import logger
from onnx import helper
from onnx import TensorProto as tp

import numpy as np

from brocolli.converter.onnx_layers.base_layer import BaseLayer


class ClipFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(ClipFunc, self).__init__(source_node, module, auto_gen)

    def generate_node(self, name=None, params=None, attr_dict=None):
        node = helper.make_node("Clip", self._in_names, self._out_names, (self._name))
        logger.info("relu6_layer: " + self._name + " created")
        self._node.append(node)

    def generate_params(self, params):
        self.create_params(self._name + "_min", params[0], tp.FLOAT)
        self.create_params(self._name + "_max", params[1], tp.FLOAT)
