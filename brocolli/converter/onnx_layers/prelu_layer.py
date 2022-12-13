from loguru import logger
import numpy as np
from onnx import helper
from onnx import TensorProto as tp


from .base_layer import BaseLayer


class PReluLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(PReluLayer, self).__init__(source_node, module, auto_gen)

    def generate_params(self, params):
        shape = self._output_shape[0]
        param_shape = [1] * len(shape)
        param_shape[1] = params.shape[0]
        params = params.reshape(param_shape)

        self.create_params(self._name + "_prelu", params)

    def generate_node(self, name=None, params=None, attr_dict=None):
        if params is None:
            weight = self._module.weight.detach().numpy()
            shape = self._output_shape[0]
            param_shape = [1] * len(shape)
            param_shape[1] = weight.shape[0]
            params = weight.reshape(param_shape)

        self.create_params(self._name + "_prelu", params)
        node = helper.make_node("PRelu", self._in_names, self._out_names, self._name)

        logger.info("prelu_layer: " + self._name + " created")
        self._node.append(node)
