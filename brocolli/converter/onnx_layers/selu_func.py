from loguru import logger
from onnx import helper

from .base_layer import BaseLayer


class SeluFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(SeluFunc, self).__init__(source_node, module, auto_gen)

    def get_relu_attr(self):
        attr_dict = {"alpha": 1.67326, "beta": 1.0507}

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):
        attr_dict = self.get_relu_attr()
        node = helper.make_node("Selu", self._in_names, self._out_names, self._name)
        logger.info("selu_layer: " + self._name + " created")
        self._node.append(node)
