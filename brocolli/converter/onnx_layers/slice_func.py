from loguru import logger
from onnx import helper
from onnx import TensorProto as tp


from .base_layer import BaseLayer


class SliceFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(SliceFunc, self).__init__(source_node, module, auto_gen)

    def generate_node(self, name=None, params=None, attr_dict=None):
        if name is not None:
            self._name = name

        self.create_params(self._name + "_start", params[0])
        self.create_params(self._name + "_end", params[1])
        self.create_params(self._name + "_axes", params[2])
        self.create_params(self._name + "_steps", params[3])

        node = helper.make_node("Slice", self._in_names, self._out_names, self._name)
        logger.info("slice_layer: " + self._name + " created")
        self._node.append(node)
