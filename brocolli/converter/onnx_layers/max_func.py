from loguru import logger
from onnx import helper

from .base_layer import BaseLayer


class MaxFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(MaxFunc, self).__init__(source_node, module, auto_gen)

    def get_mean_attr(self):
        attr_dict = {"keepdims": 1}

        attr_dict["keepdims"] = self.get_value_by_key_or_index("keepdim", 2, False)

        dim = self.get_value_by_key_or_index("dim", 1, [1])

        if isinstance(dim, int):
            dim = [dim]

        attr_dict["axes"] = dim

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):
        attr_dict = self.get_mean_attr()
        self._out_names[0] = self._name + "_0"
        node = helper.make_node(
            "ReduceMax", self._in_names, self._out_names, self._name, **attr_dict
        )

        logger.info("max_layer: " + self._name + " created")
        self._node.append(node)
