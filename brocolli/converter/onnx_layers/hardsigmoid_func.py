from loguru import logger
from onnx import helper

from .base_layer import BaseLayer


class HardsigmoidFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(HardsigmoidFunc, self).__init__(source_node, module, auto_gen)

    def get_hardsigmoid_attr(self):
        attr_dict = {"alpha": 1.0 / 6}

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):
        attr_dict = self.get_hardsigmoid_attr()
        node = helper.make_node(
            "HardSigmoid", self._in_names, self._out_names, self._name, **attr_dict
        )
        logger.info("hardsigmoid_layer: " + self._name + " created")
        self._node.append(node)
