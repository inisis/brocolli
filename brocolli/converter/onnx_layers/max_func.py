from loguru import logger
from onnx import helper

from brocolli.converter.onnx_layers.base_layer import BaseLayer


class MaxFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(MaxFunc, self).__init__(source_node, module, auto_gen)

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
        self._out_names[0] = self._name + "_0"
        node = helper.make_node(
            "ReduceMax", self._in_names, self._out_names, self._name, **attr_dict
        )

        logger.info("max_layer: " + self._name + " created")
        self._node.append(node)
