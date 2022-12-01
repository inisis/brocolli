from loguru import logger
from onnx import helper
from onnx import TensorProto as tp
import torch.nn as nn

from .base_layer import BaseLayer


class SoftmaxLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(SoftmaxLayer, self).__init__(source_node, module, auto_gen)

    def generate_node(self, name=None, params=None, attr_dict=None):
        dim = self._module.dim
        if dim is None:
            stacklevel = 3
            dim = nn.functional._get_softmax_dim(
                "softmax",
                len(self._input_shape[0]),
                stacklevel,
            )

        node = helper.make_node(
            "Softmax", self._in_names, self._out_names, self._name, axis=dim
        )
        logger.info("softmax_layer: " + self._name + " created")
        self._node.append(node)
