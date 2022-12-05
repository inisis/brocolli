from loguru import logger
from onnx import helper
from onnx import TensorProto as tp


from .base_layer import BaseLayer


class ConcatFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(ConcatFunc, self).__init__(source_node, module, auto_gen)

    def get_concat_attr(self):
        attr_dict = {"axis": []}
        dim = self.get_value_by_key_or_index("dim", 1, 0)

        attr_dict["axis"] = dim

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):
        if name is not None:
            self._name = name

        if attr_dict is None:
            attr_dict = self.get_concat_attr()
        node = helper.make_node(
            "Concat", self._in_names, self._out_names, self._name, **attr_dict
        )

        logger.info("concat_layer: " + self._name + " created")
        self._node.append(node)
