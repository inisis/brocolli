from loguru import logger
import numpy as np
from onnx import helper
from onnx import TensorProto as tp


from .base_layer import BaseLayer


class FlattenFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(FlattenFunc, self).__init__(source_node, module, auto_gen)

    def generate_node(self, name=None, params=None, attr_dict=None):
        params = np.array(self._output_shape[0], dtype=np.int64)

        self.create_params(self._name + "_flatten", params)

        node = helper.make_node("Reshape", self._in_names, self._out_names, self._name)

        logger.info("flatten_layer: " + self._name + " created")
        self._node.append(node)
