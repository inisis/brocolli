from loguru import logger
from onnx import helper

from .base_layer import BaseLayer


class TransposeFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(TransposeFunc, self).__init__(source_node, module, auto_gen)

    def gen_transpose_attr(self):
        attr_dict = {"perm": []}
        input_dim = len(self._output_shape[0])
        axes = list(range(input_dim))
        axes[self._source_node.args[1]], axes[self._source_node.args[2]] = (
            axes[self._source_node.args[2]],
            axes[self._source_node.args[1]],
        )

        attr_dict["perm"] = axes

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
