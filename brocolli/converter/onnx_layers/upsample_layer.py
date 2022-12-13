from loguru import logger
from onnx import helper
from onnx import TensorProto as tp
import numpy as np

from .base_layer import BaseLayer


class UpsampleLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(UpsampleLayer, self).__init__(source_node, module, auto_gen)

    def generate_node(self, name=None, params=None, attr_dict=None):
        if self._module.scale_factor is None:
            size = self._module.size
            if isinstance(size, int):
                dim = len(self._output_shape[0])
                output_size = self._output_shape[0]
                input_size = self._input_shape[0]
                size = [size] * len(dim)

            scales = [
                1.0
                if i < 2
                else float(output_size[-(dim - i)]) / float(input_size[-(dim - i)])
                for i in range(0, dim)
            ]
        else:
            scale_factor = self._module.scale_factor
            if isinstance(scale_factor, float):
                dim = self._output_shape[0][2:]
                scale_factor = [scale_factor] * len(dim)

            scales = [1, 1] + scale_factor

        scales = np.array(scales, dtype="float32")
        self.create_params(self._name + "_roi", np.array([]))
        self.create_params(self._name + "_scale", scales)
        node = helper.make_node(
            "Resize", self._in_names, self._out_names, self._name, mode="nearest"
        )
        logger.info("upsample_layer: " + self._name + " created")
        self._node.append(node)
