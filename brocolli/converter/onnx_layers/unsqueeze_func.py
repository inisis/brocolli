from loguru import logger
import numpy as np
from onnx import helper
from onnx import TensorProto as tp


from brocolli.converter.onnx_layers.base_layer import BaseLayer


class UnsqueezeFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(UnsqueezeFunc, self).__init__(source_node, module, auto_gen)

    def generate_node(self, name=None, params=None, attr_dict=None):
        axes = self._source_node.args[1]
        params = np.array([axes])

        self.create_params(self._name + "_unsqueeze", params, tp.INT64)
        node = helper.make_node(
            "Unsqueeze", self._in_names, self._out_names, self._name
        )

        logger.info("unsqueeze_layer: " + self._name + " created")
        self._node.append(node)
