from loguru import logger
import numpy as np
from onnx import helper
from onnx import TensorProto as tp


from brocolli.converter.onnx_layers.base_layer import BaseLayer


class TileFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(TileFunc, self).__init__(source_node, module, auto_gen)

    def generate_node(self, name=None, params=None, attr_dict=None):
        if params is None:
            if "dims" in self._source_node.kwargs:
                repeats = self._source_node.kwargs["dims"]
            elif isinstance(self._source_node.args[1], int):
                repeats = []
                for arg in self._source_node.args:
                    if isinstance(arg, int):
                        repeats.append(arg)
            else:
                repeats = self._source_node.args[1]

            params = np.array(repeats)

        self.create_params(self._name + "_tile", params, tp.INT64)

        node = helper.make_node("Tile", self._in_names, self._out_names, self._name)

        logger.info("tile_layer: " + self._name + " created")
        self._node.append(node)
