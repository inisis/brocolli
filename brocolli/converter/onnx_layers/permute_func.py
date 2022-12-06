from loguru import logger
from onnx import helper

from .base_layer import BaseLayer


class PermuteFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(PermuteFunc, self).__init__(source_node, module, auto_gen)

    def gen_transpose_attr(self):
        attr_dict = {"perm": []}
        order = []
        for arg in self._source_node.args:
            if isinstance(arg, int):
                order.append(arg)
            elif isinstance(arg, (list, tuple)):
                order.extend(arg)

        attr_dict["perm"] = order

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):
        if name is not None:
            self._name = name

        if attr_dict is None:
            attr_dict = self.gen_transpose_attr()

        node = helper.make_node(
            "Transpose", self._in_names, self._out_names, self._name, **attr_dict
        )
        logger.info("transpose_layer: " + self._name + " created")
        self._node.append(node)
