import re
from loguru import logger
import numpy as np
from onnx import helper
from onnx import TensorProto as tp

from .base_layer import BaseLayer


class ReshapeFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(ReshapeFunc, self).__init__(source_node, module, auto_gen)

    def add_bottom_top(self, in_names=None, out_names=None):
        if in_names is None:
            in_names = [self.recursive_find_name(self._source_node.args[0])]

        if out_names is None:
            out_names = [self._source_node.name]
        self._in_names.extend(in_names)
        self._out_names.extend(out_names)

    def generate_node(self, name=None, params=None, attr_dict=None):
        if name is not None:
            self._name = name

        if params is None:
            params = np.array(self._output_shape[0], dtype=np.int64)

        self.create_params(self._name + "_reshape", params)

        node = helper.make_node("Reshape", self._in_names, self._out_names, self._name)

        logger.info("reshape_layer: " + self._name + " created")
        self._node.append(node)
