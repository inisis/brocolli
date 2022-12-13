from loguru import logger
from onnx import helper
from torch.fx.node import Node
import numbers
import numpy as np

from .base_layer import BaseLayer
from ..utils import torch_dtype_to_numpy


class MulLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(MulLayer, self).__init__(source_node, module, auto_gen)

    def generate_node(self, name=None, params=None, attr_dict=None):
        if self._auto_gen:
            assert len(self._source_node.args) == 2
            if len(self._source_node.all_input_nodes) == 1:
                if isinstance(self._source_node.args[0], Node):
                    assert isinstance(self._source_node.args[1], numbers.Number)
                    if self._output_dtype:
                        numpy_dtype = torch_dtype_to_numpy(self._output_dtype[0])
                        self.generate_params(
                            np.array([self._source_node.args[1]], dtype=numpy_dtype)
                        )
                    else:
                        self.generate_params(np.array([self._source_node.args[1]]))
                else:
                    assert isinstance(self._source_node.args[0], numbers.Number)
                    if self._output_dtype:
                        numpy_dtype = torch_dtype_to_numpy(self._output_dtype[0])
                        self.generate_params(
                            np.array([self._source_node.args[0]], dtype=numpy_dtype)
                        )
                    else:
                        self.generate_params(np.array([self._source_node.args[0]]))

            node = helper.make_node("Mul", self._in_names, self._out_names, self._name)

            logger.info("mul_layer: " + self._name + " created")
            self._node.append(node)
        else:
            mul_layer = MulFunc(self._source_node, self._module, auto_gen=False)
            mul_layer.add_bottom_top(in_names=self._in_names)
            mul_layer.generate_node(self._name)
            self.node_post_process(mul_layer)

    def generate_params(self, params, dytpe=None):
        self.create_params(self._name + "_mul_constant", params, dytpe)


class MulFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(MulFunc, self).__init__(source_node, module, auto_gen)

    def generate_node(self, name=None, params=None, attr_dict=None):
        node = helper.make_node("Mul", self._in_names, self._out_names, self._name)

        logger.info("mul_layer: " + self._name + " created")
        self._node.append(node)

    def generate_params(self, params, dtype=None):
        self.create_params(self._name + "_mul_constant", params, dtype)
