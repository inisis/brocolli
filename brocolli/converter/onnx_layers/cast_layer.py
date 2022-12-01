from loguru import logger
from onnx import helper

from .base_layer import BaseLayer


class CastLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(CastLayer, self).__init__(source_node, module, auto_gen)

    def get_cast_attr(self):
        attr_dict = {"to": 1}

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):
        if name is not None:
            self._name = name

        if attr_dict is None:
            attr_dict = self.get_cast_attr()

        node = helper.make_node(
            "Cast", self._in_names, self._out_names, self._name, **attr_dict
        )
        logger.info("cast_layer: " + self._name + " created")
        self._node.append(node)
