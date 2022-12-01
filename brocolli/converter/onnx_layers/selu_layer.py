from loguru import logger
import numpy as np
from onnx import helper
from onnx import TensorProto as tp

from .base_layer import BaseLayer


class SeluLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(SeluLayer, self).__init__(source_node, module, auto_gen)

    def get_selu_attr(self):
        attr_dict = {
            "alpha": 1.67326319217681884765625,  # float defaults is 1.67326319217681884765625
            "gamma": 1.05070102214813232421875,  # float default is 1.05070102214813232421875
        }

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):
        attr_dict = self.get_selu_attr()
        logger.debug(attr_dict)
        node = helper.make_node(
            "Selu", self._in_names, self._out_names, self._name, **attr_dict
        )
        logger.info("selu_layer: " + self._name + " created")
        self._node.append(node)
