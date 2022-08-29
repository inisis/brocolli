from loguru import logger
from onnx import helper
from onnx import TensorProto as tp
from torch.fx.immutable_collections import immutable_list
from onnx_layers.base_layer import BaseLayer


class InputLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(InputLayer, self).__init__(source_node, module, auto_gen)
        self._generate_input()

    def _generate_input(self):
        if type(self.tensor_meta) is tuple or type(self.tensor_meta) is immutable_list:
            for idx, tensor_meta in enumerate(self.tensor_meta):
                input_tvi = helper.make_tensor_value_info(
                    self._name + "_" + str(idx), tp.FLOAT, tensor_meta.shape
                )
                logger.info("input_layer: " + self._name + "_" + str(idx) + " created")
                self._in_tensor_value_info.append(input_tvi)
        else:
            input_tvi = helper.make_tensor_value_info(
                self._name, tp.FLOAT, self.tensor_meta.shape
            )
            logger.info("input_layer: " + self._name + " created")
            self._in_tensor_value_info.append(input_tvi)
