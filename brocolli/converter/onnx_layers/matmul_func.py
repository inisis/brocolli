from loguru import logger
from onnx import helper
from onnx import TensorProto as tp

from .base_layer import BaseLayer


class MatmulFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(MatmulFunc, self).__init__(source_node, module, auto_gen)

    def generate_node(self, name=None, params=None, attr_dict=None):
        if name is not None:
            self._name = name

        node = helper.make_node("MatMul", self._in_names, self._out_names, self._name)

        logger.info("matmul_layer: " + self._name + " created")
        self._node.append(node)

    def generate_params(self, params):
        self.create_params(self._name + "_matmul_constant", params)
