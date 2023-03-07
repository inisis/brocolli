import numpy as np

from .base_layer import BaseLayer
from .split_func import SplitFunc
from .sigmoid_func import SigmoidFunc
from .mul_layer import MulFunc


class GLUFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(GLUFunc, self).__init__(source_node, module, auto_gen)

    def add_bottom_top(self, in_names=None, out_names=None):
        pass

    def generate_node(self, name=None, params=None, attr_dict=None):
        axis = self.get_value_by_key_or_index("dim", 0)
        split_shape = self._output_shape[0][axis]
        split_layer = SplitFunc(self._source_node, auto_gen=False)
        split_layer.add_bottom_top(
            out_names=[
                self._source_node.name + "_split_0",
                self._source_node.name + "_split_1",
            ]
        )
        params_split = np.array([split_shape] * 2)
        attr_dict = {"axis": axis}
        split_layer.generate_node(
            self._source_node.name + "_split", params_split, attr_dict
        )
        self.node_post_process(split_layer)

        sigmoid_layer = SigmoidFunc(self._source_node, auto_gen=False)
        sigmoid_layer.add_bottom_top(
            in_names=[self._source_node.name + "_split_1"],
            out_names=[self._source_node.name + "_sigmoid"],
        )
        sigmoid_layer.generate_node(self._source_node.name + "_sigmoid")
        self.node_post_process(sigmoid_layer)

        mul_layer = MulFunc(self._source_node, auto_gen=False)
        mul_layer.add_bottom_top(
            in_names=[
                self._source_node.name + "_split_0",
                self._source_node.name + "_sigmoid",
            ]
        )
        mul_layer.generate_node(self._source_node.name + "_mul")
        self.node_post_process(mul_layer)
