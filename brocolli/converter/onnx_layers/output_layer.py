from loguru import logger
from onnx import helper
from onnx import TensorProto as tp
import torch

from .base_layer import BaseLayer
from ..utils import pytorch_dtype_to_onnx


class OutputLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(OutputLayer, self).__init__(source_node, module, auto_gen)
        self.generate_output()

    def add_bottom_top(self, in_names=None, out_names=None):
        pass

    def generate_output(self):
        if self._output_type is torch.Tensor or len(self._output_shape) == 1:
            name = self.recursive_find_name(self._source_node.all_input_nodes[0])
            onnx_type = pytorch_dtype_to_onnx(self._output_dtype[0])
            output_tvi = helper.make_tensor_value_info(
                name, onnx_type, self._output_shape[0]
            )
            logger.info("output_layer: " + name + " created")
            self._out_tensor_value_info.append(output_tvi)
        elif len(self._output_shape) == len(self._source_node.all_input_nodes):
            for idx, shape in enumerate(self._output_shape):
                name = self.recursive_find_name(self._source_node.all_input_nodes[idx])
                onnx_type = pytorch_dtype_to_onnx(self._output_dtype[idx])
                output_tvi = helper.make_tensor_value_info(name, onnx_type, shape)
                logger.info("output_layer: " + name + " created")
                self._out_tensor_value_info.append(output_tvi)
        else:
            for idx, shape in enumerate(self._output_shape):
                name = self.recursive_find_name(self._source_node.all_input_nodes[0])
                onnx_type = pytorch_dtype_to_onnx(self._output_dtype[idx])
                output_tvi = helper.make_tensor_value_info(
                    name + "_" + str(idx), onnx_type, shape
                )
                logger.info(
                    "output_layer: "
                    + self._source_node.all_input_nodes[0].name
                    + "_"
                    + str(idx)
                    + " created"
                )
                self._out_tensor_value_info.append(output_tvi)
