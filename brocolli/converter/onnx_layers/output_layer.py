from loguru import logger
from onnx import helper
from onnx import TensorProto as tp
import torch

from brocolli.converter.onnx_layers.base_layer import BaseLayer


class OutputLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(OutputLayer, self).__init__(source_node, module, auto_gen)
        if self._auto_gen:
            output_name = self.recursive_find_name(self._source_node.args[0])
            self.generate_output(output_name)

    def add_bottom_top(self, in_names=None, out_names=None):
        pass

    def generate_output(self, name):
        if self._output_type is torch.Tensor:
            output_tvi = helper.make_tensor_value_info(
                name, tp.FLOAT, self._output_shape[0]
            )
            logger.info("output_layer: " + name + " created")
            self._out_tensor_value_info.append(output_tvi)
        else:
            for idx, shape in enumerate(self._output_shape):
                output_tvi = helper.make_tensor_value_info(
                    name + "_" + str(idx), tp.FLOAT, shape
                )
                logger.info(
                    "output_layer: "
                    + self._source_node.args[0].name
                    + "_"
                    + str(idx)
                    + " created"
                )
                self._out_tensor_value_info.append(output_tvi)
