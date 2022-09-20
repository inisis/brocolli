from loguru import logger
from onnx import helper
from onnx import TensorProto as tp
import onnx_layers as ops
import numpy as np

from brocolli.converter.onnx_layers.base_layer import BaseLayer


class ChunkFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(ChunkFunc, self).__init__(source_node, module, auto_gen)

    def add_bottom_top(self, in_names=None, out_names=None):
        pass

    def generate_node(self, name=None, params=None, attr_dict=None):
        if "dim" in self._source_node.kwargs:
            axis = self._source_node.kwargs["dim"]
        else:
            axis = self._source_node.args[2]

        sum_ = 0
        for idx in range(len(self._input_shape)):
            slice_shape = self._input_shape[idx][axis]
            slice_layer = ops.SliceFunc(self._source_node, auto_gen=False)
            slice_layer.add_bottom_top(
                out_names=[self._source_node.name + "_" + str(idx)]
            )
            params_slice = [
                np.array([sum_]),
                np.array([sum_ + slice_shape]),
                np.array([axis]),
                np.array([1]),
            ]
            slice_layer.generate_node(
                self._source_node.name + "_" + str(idx), params_slice
            )
            self.node_post_process(slice_layer)
            sum_ += slice_shape
