from loguru import logger
from onnx import helper
from onnx import TensorProto as tp
import torch
from brocolli.converter.onnx_layers.base_layer import BaseLayer


class InputLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(InputLayer, self).__init__(source_node, module, auto_gen)
        self._generate_input()

    def _generate_input(self):
        if self._output_type is not torch.Tensor:
            for idx in range(len(self._output_shape)):
                input_tvi = helper.make_tensor_value_info(
                    self._name + "_" + str(idx), tp.FLOAT, self._output_shape[idx]
                )
                logger.info("input_layer: " + self._name + "_" + str(idx) + " created")
                self._in_tensor_value_info.append(input_tvi)
        else:
            input_tvi = helper.make_tensor_value_info(
                self._name, tp.FLOAT, self._output_shape[0]
            )
            logger.info("input_layer: " + self._name + " created")
            self._in_tensor_value_info.append(input_tvi)
