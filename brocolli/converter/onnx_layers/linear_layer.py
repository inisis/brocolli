from loguru import logger
import numpy as np

from .base_layer import BaseLayer
from .gemm_layer import GemmLayer
from .reshape_func import ReshapeFunc
from .matmul_func import MatmulFunc
from .add_layer import AddLayer


class LinearLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(LinearLayer, self).__init__(source_node, module, auto_gen)

    def add_bottom_top(self, in_names=None, out_names=None):
        pass

    def generate_node(self, name=None, params=None, attr_dict=None):
        if len(self._output_shape[0]) == 2:
            gemm_layer = GemmLayer(self._source_node, self._module)
            self.node_post_process(gemm_layer)
        elif len(self._output_shape[0]) >= 2 and self._output_shape[0][2:] == [1] * (
            len(self._output_shape[0]) - 2
        ):
            reshape_layer = ReshapeFunc(self._source_node, self._module, auto_gen=False)
            reshape_layer.add_bottom_top(
                out_names=[self._source_node.name + "_reshape"]
            )
            params = np.array([1, -1])
            reshape_layer.generate_node(
                self._source_node.name + "_reshape", params=params
            )
            self.node_post_process(reshape_layer)

            gemm_layer = GemmLayer(self._source_node, self._module, auto_gen=False)
            gemm_layer.add_bottom_top(in_names=[self._source_node.name + "_reshape"])
            gemm_layer.generate_node(self._source_node.name)
            self.node_post_process(gemm_layer)
        else:
            if self._module.bias is not None:
                matmul_layer = MatmulFunc(
                    self._source_node, self._module, auto_gen=False
                )
                matmul_layer.add_bottom_top(
                    out_names=[self._source_node.name + "_Matmul"]
                )
                matmul_layer.generate_params(
                    self._module.weight.transpose(1, 0).detach().numpy()
                )
                matmul_layer.generate_node(self._source_node.name + "_mul")
                self.node_post_process(matmul_layer)

                add_layer = AddLayer(self._source_node, self._module, auto_gen=False)
                add_layer.add_bottom_top(in_names=[self._source_node.name + "_Matmul"])
                add_layer.generate_node(
                    self._source_node.name, params=self._module.bias.detach().numpy()
                )
                self.node_post_process(add_layer)
            else:
                matmul_layer = MatmulFunc(
                    self._source_node, self._module, auto_gen=False
                )
                matmul_layer.add_bottom_top()
                matmul_layer.generate_params(
                    self._module.weight.transpose(1, 0).detach().numpy()
                )
                matmul_layer.generate_node(self._source_node.name)
                self.node_post_process(matmul_layer)
