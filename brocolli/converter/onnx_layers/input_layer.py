from loguru import logger
from onnx import helper
import torch
from .base_layer import BaseLayer
from ..utils import pytorch_dtype_to_onnx


class InputLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(InputLayer, self).__init__(source_node, module, auto_gen)
        self._generate_input()

    def _generate_input(self):
        if self._output_type is not torch.Tensor:
            for idx in range(len(self._output_shape)):
                onnx_type = pytorch_dtype_to_onnx(self._output_dtype[idx])
                input_tvi = helper.make_tensor_value_info(
                    self._name + "_" + str(idx), onnx_type, self._output_shape[idx]
                )
                logger.info("input_layer: " + self._name + "_" + str(idx) + " created")
                self._in_tensor_value_info.append(input_tvi)
        else:
            onnx_type = pytorch_dtype_to_onnx(self._output_dtype[0])
            input_tvi = helper.make_tensor_value_info(
                self._name, onnx_type, self._output_shape[0]
            )
            logger.info("input_layer: " + self._name + " created")
            self._in_tensor_value_info.append(input_tvi)
