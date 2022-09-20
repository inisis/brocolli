from loguru import logger
from onnx import helper
from onnx import TensorProto as tp

import numpy as np

from brocolli.converter.onnx_layers.base_layer import BaseLayer


class Relu6Func(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(Relu6Func, self).__init__(source_node, module, auto_gen)

    def generate_node(self, name=None, params=None, attr_dict=None):
        self.create_params(self._name + "_min", np.array(0), tp.FLOAT)
        self.create_params(self._name + "_max", np.array(6), tp.FLOAT)
        node = helper.make_node("Clip", self._in_names, self._out_names, (self._name))
        logger.info("relu6_layer: " + self._name + " created")
        self._node.append(node)
