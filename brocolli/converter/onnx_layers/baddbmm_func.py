import numpy as np

from .base_layer import BaseLayer
from .matmul_func import MatmulFunc
from .add_layer import AddFunc
from .mul_layer import MulFunc


class BADDBMMFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(BADDBMMFunc, self).__init__(source_node, module, auto_gen)

    def add_bottom_top(self, in_names=None, out_names=None):
        pass

    def generate_node(self, name=None, params=None, attr_dict=None):
        matmul_layer = MatmulFunc(self._source_node, auto_gen=False)
        matmul_layer.add_bottom_top(
            in_names=[
                self.recursive_find_name(self._source_node.args[1]),
                self.recursive_find_name(self._source_node.args[2]),
            ],
            out_names=[self._source_node.name + "_matmul"],
        )
        matmul_layer.generate_node(self._source_node.name + "_matmul")
        self.node_post_process(matmul_layer)

        mul_layer = MulFunc(self._source_node, auto_gen=False)
        params = np.array(
            [self.get_value_by_key_or_index("alpha", 4, 1.0)], dtype=np.float32
        )
        self.create_params(self._source_node.name + "_mul_constant", params)
        mul_layer.add_bottom_top(
            in_names=[
                self._source_node.name + "_matmul",
                self._source_node.name + "_mul_constant",
            ],
            out_names=[self._source_node.name + "_mul"],
        )
        mul_layer.generate_node(self._source_node.name + "_mul")
        self.node_post_process(mul_layer)

        mul_layer_1 = MulFunc(self._source_node, auto_gen=False)
        params = np.array(
            [self.get_value_by_key_or_index("beta", 3, 1.0)], dtype=np.float32
        )
        self.create_params(self._source_node.name + "_mul_constant_1", params)
        mul_layer_1.add_bottom_top(
            in_names=[
                self.recursive_find_name(self._source_node.args[0]),
                self._source_node.name + "_mul_constant_1",
            ],
            out_names=[self._source_node.name + "_mul_1"],
        )
        mul_layer_1.generate_node(self._source_node.name + "_mul_1")
        self.node_post_process(mul_layer_1)

        add_layer = AddFunc(self._source_node, auto_gen=False)
        add_layer.add_bottom_top(
            in_names=[
                self._source_node.name + "_mul",
                self._source_node.name + "_mul_1",
            ]
        )
        add_layer.generate_node(self._source_node.name + "_add")
        self.node_post_process(add_layer)
