from loguru import logger
import numpy as np
from onnx import helper
from onnx import TensorProto as tp


from .base_layer import BaseLayer
from .unsqueeze_func import UnsqueezeFunc
from .concat_func import ConcatFunc


class StackFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(StackFunc, self).__init__(source_node, module, auto_gen)

    def add_bottom_top(self, in_names=None, out_names=None):
        pass

    def generate_node(self, name=None, params=None, attr_dict=None):
        dim = self.get_value_by_key_or_index("dim", 1, 0)
        concat_in_names = []
        for idx in range(len(self._input_shape)):
            unsqueeze_layer = UnsqueezeFunc(
                self._source_node, self._module, auto_gen=False
            )
            unsqueeze_layer.add_bottom_top(
                in_names=[self.recursive_find_name(self._source_node.args[0][idx])],
                out_names=[self._source_node.name + "_unsqueeze_" + str(idx)],
            )
            params = np.array(dim)
            unsqueeze_layer.generate_node(
                self._source_node.name + "_unsqueeze_" + str(idx), params=params
            )
            concat_in_names.append(self._source_node.name + "_unsqueeze_" + str(idx))
            self.node_post_process(unsqueeze_layer)

        concat_layer = ConcatFunc(self._source_node, self._module, auto_gen=False)
        concat_layer.add_bottom_top(in_names=concat_in_names)
        concat_layer.generate_node(self._source_node.name)
        self.node_post_process(concat_layer)
