from loguru import logger
from onnx import helper
from onnx import TensorProto as tp


from .base_layer import BaseLayer


class EmbeddingLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(EmbeddingLayer, self).__init__(source_node, module, auto_gen)

    def get_gather_attr(self):
        attr_dict = {"axis": 0}

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):
        if name is not None:
            self._name = name

        if attr_dict is None:
            attr_dict = self.get_gather_attr()

        self.create_params(self._name + "_weight", self._module.weight.detach().numpy())

        self._in_names.reverse()  # weight & input

        node = helper.make_node(
            "Gather", self._in_names, self._out_names, self._name, **attr_dict
        )

        logger.info("gather_layer: " + self._name + " created")
        self._node.append(node)
