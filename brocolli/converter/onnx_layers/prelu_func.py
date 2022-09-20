from loguru import logger
import numpy as np
from onnx import helper
from onnx import TensorProto as tp


from brocolli.converter.onnx_layers.base_layer import BaseLayer


class PReluFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(PReluFunc, self).__init__(source_node, module, auto_gen)

    def generate_node(self, name=None, params=None, attr_dict=None):
        node = helper.make_node("PRelu", self._in_names, self._out_names, self._name)

        logger.info("prelu_layer: " + self._name + " created")
        self._node.append(node)

    def generate_params(self, params):
        shape = self._output_shape[0]
        param_shape = [1] * len(shape)
        param_shape[1] = params.shape[0]
        params = params.reshape(param_shape)

        self.create_params(self._name + "_prelu", params, tp.FLOAT)
