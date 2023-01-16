from loguru import logger
from onnx import helper
from onnx import TensorProto as tp
import numpy as np

from .base_layer import BaseLayer


class UpsampleFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(UpsampleFunc, self).__init__(source_node, module, auto_gen)

    def generate_node(self, name=None, params=None, attr_dict=None):
        scale_factor = self._source_node.kwargs["scale_factor"]

        if scale_factor is None:
            dim = len(self._output_shape[0])
            output_size = self._output_shape[0]
            input_size = self._input_shape[0]

            scales = [
                1.0
                if i < 2
                else float(output_size[-(dim - i)]) / float(input_size[-(dim - i)])
                for i in range(0, dim)
            ]
        else:
            if isinstance(scale_factor, float):
                dim = self._output_shape[0][2:]
                scale_factor = [scale_factor] * len(dim)

            scales = [1, 1] + scale_factor

        scales = np.array(scales, dtype="float32")
        self.create_params(self._name + "_roi", np.array([], dtype=np.float32))
        self.create_params(self._name + "_scale", scales)
        node = helper.make_node(
            "Resize", self._in_names, self._out_names, self._name, mode="nearest"
        )
        logger.info("upsample_layer: " + self._name + " created")
        self._node.append(node)
