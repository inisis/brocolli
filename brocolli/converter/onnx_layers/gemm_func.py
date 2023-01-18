from loguru import logger
import numpy as np
from onnx import helper
from onnx import TensorProto as tp

from .base_layer import BaseLayer


class GemmFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(GemmFunc, self).__init__(source_node, module, auto_gen)

    def get_gemm_attr(self):
        attr_dict = {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 1}

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):
        if name is not None:
            self._name = name

        attr_dict = self.get_gemm_attr()

        logger.debug(attr_dict)
        node = helper.make_node(
            "Gemm", self._in_names, self._out_names, self._name, **attr_dict
        )
        logger.info("gemm_layer: " + self._name + " created")
        self._node.append(node)
