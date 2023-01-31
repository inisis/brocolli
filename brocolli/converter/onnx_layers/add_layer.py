from loguru import logger
from onnx import helper
from torch.fx.node import Node

import numbers
import numpy as np
import torch

from .base_layer import BaseLayer
from .cast_layer import CastLayer
from ..utils import torch_dtype_to_numpy


class AddLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(AddLayer, self).__init__(source_node, module, auto_gen)

    def generate_node(self, name=None, params=None, attr_dict=None):
        if name is not None:
            self._name = name

        if self._auto_gen:
            assert len(self._source_node.args) == 2
            if len(self._source_node.all_input_nodes) == 1:
                add_layer = AddFunc(self._source_node)
                self.node_post_process(add_layer)
            else:
                if self._input_dtype[0] == self._input_dtype[1]:
                    add_layer = AddFunc(self._source_node)
                    self.node_post_process(add_layer)
                elif self._input_dtype[0] != torch.float32:
                    cast_layer = CastLayer(
                        self._source_node, self._module, auto_gen=False
                    )
                    cast_layer.add_bottom_top(
                        in_names=[self.recursive_find_name(self._source_node.args[0])],
                        out_names=[self._source_node.name + "_cast"],
                    )
                    cast_layer.generate_node(
                        self._source_node.name + "_cast", attr_dict={"to": 1}
                    )
                    self.node_post_process(cast_layer)

                    add_layer = AddFunc(self._source_node, self._module, auto_gen=False)
                    add_layer.add_bottom_top(
                        in_names=[
                            self._source_node.name + "_cast",
                            self.recursive_find_name(self._source_node.args[1]),
                        ]
                    )
                    add_layer.generate_node(self._source_node.name)
                    self.node_post_process(add_layer)
                elif self._input_dtype[1] != torch.float32:
                    cast_layer = CastLayer(
                        self._source_node, self._module, auto_gen=False
                    )
                    cast_layer.add_bottom_top(
                        in_names=[self.recursive_find_name(self._source_node.args[1])],
                        out_names=[self._source_node.name + "_cast"],
                    )
                    cast_layer.generate_node(
                        self._source_node.name + "_cast", attr_dict={"to": 1}
                    )
                    self.node_post_process(cast_layer)

                    add_layer = AddFunc(self._source_node, self._module, auto_gen=False)
                    add_layer.add_bottom_top(
                        in_names=[
                            self.recursive_find_name(self._source_node.args[0]),
                            self._source_node.name + "_cast",
                        ]
                    )
                    add_layer.generate_node(self._source_node.name)
                    self.node_post_process(add_layer)
        else:
            add_layer = AddFunc(self._source_node, self._module, auto_gen=False)
            add_layer.add_bottom_top(in_names=self._in_names)
            add_layer.generate_params(params)
            add_layer.generate_node(self._name)
            self.node_post_process(add_layer)


class AddFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(AddFunc, self).__init__(source_node, module, auto_gen)

    def generate_node(self, name=None, params=None, attr_dict=None):
        if name is not None:
            self._name = name

        if self._auto_gen:
            assert len(self._source_node.args) == 2
            if len(self._source_node.all_input_nodes) == 1:
                if isinstance(self._source_node.args[0], Node):
                    assert isinstance(self._source_node.args[1], numbers.Number)
                    if self._output_dtype:
                        numpy_dtype = torch_dtype_to_numpy(self._output_dtype[0])
                        self.generate_params(
                            np.array(self._source_node.args[1], dtype=numpy_dtype)
                        )
                    else:
                        self.generate_params(np.array([self._source_node.args[1]]))
                else:
                    assert isinstance(self._source_node.args[0], numbers.Number)
                    if self._output_dtype:
                        numpy_dtype = torch_dtype_to_numpy(self._output_dtype[0])
                        self.generate_params(
                            np.array(self._source_node.args[0], dtype=numpy_dtype)
                        )
                    else:
                        self.generate_params(np.array(self._source_node.args[0]))

        node = helper.make_node("Add", self._in_names, self._out_names, self._name)
        logger.info("add_layer: " + self._name + " created")
        self._node.append(node)

    def generate_params(self, params, dtype=None):
        self.create_params(self._name + "_add_constant", params, dtype)
