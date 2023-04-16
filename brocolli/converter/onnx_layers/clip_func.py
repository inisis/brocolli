from onnx import helper
from loguru import logger
from torch.fx.node import Node

import numpy as np

from .base_layer import BaseLayer


class ClipFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(ClipFunc, self).__init__(source_node, module, auto_gen)

    def generate_node(self, name=None, params=None, attr_dict=None):
        if isinstance(self._source_node.kwargs["min"], Node) and isinstance(
            self._source_node.kwargs["max"], Node
        ):
            self._in_names.append(self._source_node.kwargs["min"].name)
            self._in_names.append(self._source_node.kwargs["max"].name)
        else:
            params_clip = [
                np.array(self._source_node.kwargs["min_val"]),
                np.array(self._source_node.kwargs["max_val"]),
            ]
            self.generate_params(params_clip)
        node = helper.make_node("Clip", self._in_names, self._out_names, (self._name))
        logger.info(f"{self.__class__.__name__}: {self._name} created")
        self._node.append(node)

    def generate_params(self, params):
        self.create_params(self._name + "_min", params[0])
        logger.debug(
            f"relu6_layer: {self._name} create param: {self._name}_min, value: {params[0]}"
        )
        self.create_params(self._name + "_max", params[1])
        logger.debug(
            f"relu6_layer: {self._name} create param: {self._name}_max, value: {params[1]}"
        )
