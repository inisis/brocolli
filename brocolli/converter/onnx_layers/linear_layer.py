from loguru import logger
from onnx import helper
from onnx import TensorProto as tp
import onnx_layers as ops
import numpy as np

from onnx_layers.base_layer import BaseLayer


class LinearLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(LinearLayer, self).__init__(source_node, module, auto_gen)

    def add_bottom_top(self, in_names=None, out_names=None):
        pass

    def generate_node(self, name=None, params=None, attr_dict=None):
        if len(self._output_shape) == 2:
            gemm_layer = ops.GemmLayer(self._source_node, self._module)
            self.node_post_process(gemm_layer)
        else:
            reshape_layer = ops.ReshapeFunc(
                self._source_node, self._module, auto_gen=False
            )
            reshape_layer.add_bottom_top(
                out_names=[self._source_node.name + "_reshape"]
            )
            params = np.array([1, -1])
            reshape_layer.generate_node(
                self._source_node.name + "_reshape", params=params
            )
            self.node_post_process(reshape_layer)

            gemm_layer = ops.GemmLayer(self._source_node, self._module, auto_gen=False)
            gemm_layer.add_bottom_top(in_names=[self._source_node.name + "_reshape"])
            gemm_layer.generate_node(self._source_node.name)
            self.node_post_process(gemm_layer)
