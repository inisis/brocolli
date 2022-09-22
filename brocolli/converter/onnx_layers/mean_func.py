from loguru import logger
from onnx import helper
from onnx import TensorProto as tp
import numpy as np

from brocolli.converter.onnx_layers.base_layer import BaseLayer


class MeanFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(MeanFunc, self).__init__(source_node, module, auto_gen)

    def get_mean_attr(self):
        attr_dict = {"keepdims": 1}

        if "keepdim" in self._source_node.kwargs:
            attr_dict["keepdims"] = self._source_node.kwargs["keepdim"]
        else:
            attr_dict["keepdims"] = self.list_try_get(self._source_node.args, 2, False)

        if "dim" in self._source_node.kwargs:
            dim = self._source_node.kwargs["dim"]
        else:
            dim = self.list_try_get(self._source_node.args, 1, [1])

        if isinstance(dim, int):
            dim = [dim]

        attr_dict["axes"] = dim

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):
        attr_dict = self.get_mean_attr()
        node = helper.make_node(
            "ReduceMean", self._in_names, self._out_names, self._name, **attr_dict
        )

        logger.info("mean_layer: " + self._name + " created")
        self._node.append(node)
