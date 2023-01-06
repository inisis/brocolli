from loguru import logger
import numpy as np
from onnx import helper
from onnx import TensorProto as tp


from .base_layer import BaseLayer


class TileFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(TileFunc, self).__init__(source_node, module, auto_gen)

    def add_bottom_top(self, in_names=None, out_names=None):
        if in_names is None:
            in_names = [self.recursive_find_name(self._source_node.args[0])]

        if out_names is None:
            out_names = []

            if len(self._output_shape) == 1:
                out_names.append(self._name)
            else:
                for idx in range(len(self._output_shape)):
                    out_names.append(self._name + "_" + str(idx))

        self._in_names.extend(in_names)
        self._out_names.extend(out_names)

    def generate_node(self, name=None, params=None, attr_dict=None):
        if params is None:
            repeats = np.array(self._output_shape[0]) // np.array(self._input_shape[0])
            params = np.array(repeats, dtype=np.int64)

        self.create_params(self._name + "_tile", params)

        node = helper.make_node("Tile", self._in_names, self._out_names, self._name)
        logger.info("tile_layer: " + self._name + " created")
        self._node.append(node)
