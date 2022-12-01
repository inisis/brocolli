from loguru import logger
from onnx import helper

from .base_layer import BaseLayer


class CeluFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(CeluFunc, self).__init__(source_node, module, auto_gen)

    def get_celu_attr(self):
        attr_dict = {"alpha": 0}

        attr_dict["alpha"] = self._source_node.kwargs["alpha"]

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):
        attr_dict = self.get_celu_attr()
        node = helper.make_node(
            "Celu", self._in_names, self._out_names, self._name, **attr_dict
        )
        logger.info("celu_layer: " + self._name + " created")
        self._node.append(node)
