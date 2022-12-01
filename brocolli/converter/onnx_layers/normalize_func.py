from loguru import logger
from onnx import helper

from .base_layer import BaseLayer


class NormalizeFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(NormalizeFunc, self).__init__(source_node, module, auto_gen)

    def get_relu_attr(self):
        attr_dict = {"axis": -1, "p": 2}

        attr_dict["axis"] = self._source_node.kwargs["dim"]
        attr_dict["p"] = self._source_node.kwargs["p"]

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):
        attr_dict = self.get_relu_attr()
        node = helper.make_node(
            "LpNormalization", self._in_names, self._out_names, self._name, **attr_dict
        )
        logger.info("lpnormalization_layer: " + self._name + " created")
        self._node.append(node)
