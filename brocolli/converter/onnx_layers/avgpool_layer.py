from loguru import logger
import numpy as np
import torch.nn as nn

from .base_layer import BaseLayer
from .pad_layer import PadLayer
from .pooling_layer import PoolingLayer


class AvgPoolLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(AvgPoolLayer, self).__init__(source_node, module, auto_gen)

    def add_bottom_top(self, in_names=None, out_names=None):
        pass

    def generate_node(self, name=None, params=None, attr_dict=None):
        pad_layer = PadLayer(self._source_node, self._module, auto_gen=False)

        pad_layer.add_bottom_top(out_names=[self._source_node.name + "_pad"])

        if isinstance(self._module.padding, tuple):
            if len(self._module.padding) == 1:
                pad_h = pad_w = self._module.padding[0]
            else:
                pad_h = self._module.padding[0]
                pad_w = self._module.padding[1]
        else:
            pad_h = pad_w = self._module.padding

        if isinstance(self._module, nn.AvgPool1d):
            pads = [0, 0, pad_h, 0, 0, pad_h]
        elif isinstance(self._module, nn.AvgPool2d):
            pads = [0, 0, pad_h, pad_w, 0, 0, pad_h, pad_w]

        params = [np.array(pads), np.array(0.0, dtype="float32")]

        pad_layer.generate_node(
            self._source_node.name + "_pad",
            params=params,
            attr_dict={"mode": "constant"},
        )
        self.node_post_process(pad_layer)

        pooling_layer = PoolingLayer(self._source_node, self._module, auto_gen=False)
        pooling_layer.add_bottom_top(in_names=[self._source_node.name + "_pad"])
        pooling_layer.generate_node(self._source_node.name)
        self.node_post_process(pooling_layer)
