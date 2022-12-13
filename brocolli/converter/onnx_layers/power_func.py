from loguru import logger
import numpy as np
from onnx import helper
from onnx import TensorProto as tp


from .base_layer import BaseLayer


class PowerFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(PowerFunc, self).__init__(source_node, module, auto_gen)

    def generate_node(self, name=None, params=None, attr_dict=None):
        params = np.array([self._source_node.args[1]])
        self.create_params(self._name + "_weight", params)
        node = helper.make_node("Pow", self._in_names, self._out_names, self._name)
        logger.info("power_layer: " + self._name + " created")
        self._node.append(node)

    def generate_params(self, params):
        self.create_params(self._name + "_weight", params)
