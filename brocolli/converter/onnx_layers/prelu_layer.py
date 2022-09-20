from loguru import logger
import numpy as np
from onnx import helper
from onnx import TensorProto as tp


from brocolli.converter.onnx_layers.base_layer import BaseLayer


class PReluLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(PReluLayer, self).__init__(source_node, module, auto_gen)

    def create_prelu_params(self):
        param_name = self._name + "_prelu"
        params = self._module.weight.detach().numpy()

        output_shape = list(self._output_shape)

        param_type = tp.FLOAT
        param_shape = [1] * len(output_shape)
        param_shape[1] = params.shape[0]

        param_tensor_value_info = helper.make_tensor_value_info(
            param_name, param_type, param_shape
        )
        param_tensor = helper.make_tensor(
            param_name, param_type, param_shape, params.flatten()
        )
        self._in_names.append(param_name)
        self._in_tensor_value_info.append(param_tensor_value_info)
        self._init_tensor.append(param_tensor)

    def generate_node(self, name=None, params=None, attr_dict=None):
        if params is None:
            weight = self._module.weight.detach().numpy()
            shape = self._output_shape[0]
            param_shape = [1] * len(shape)
            param_shape[1] = weight.shape[0]
            params = weight.reshape(param_shape)

        self.create_params(self._name + "_prelu", params, tp.FLOAT)
        node = helper.make_node("PRelu", self._in_names, self._out_names, self._name)

        logger.info("prelu_layer: " + self._name + " created")
        self._node.append(node)
