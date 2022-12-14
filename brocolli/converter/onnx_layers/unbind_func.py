from loguru import logger
from onnx import helper
from onnx import TensorProto as tp
import numpy as np

from .base_layer import BaseLayer
from .split_func import SplitFunc
from .squeeze_func import SqueezeFunc


class UnbindFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(UnbindFunc, self).__init__(source_node, module, auto_gen)

    def get_split_attr(self):
        attr_dict = {"axis": 0}
        dim = self.get_value_by_key_or_index("dim", 1, 0)
        attr_dict["axis"] = dim

        return attr_dict

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
        split_out_names = []
        shape = np.array([1] * len(self._output_shape), dtype=np.int64)
        for idx in range(len(self._output_shape)):
            split_out_names.append(self._name + "_split_" + str(idx))
        attr_dict = self.get_split_attr()

        split_layer = SplitFunc(self._source_node, auto_gen=False)
        split_layer.add_bottom_top(out_names=split_out_names)
        split_layer.generate_node(self._name + "_split", shape, attr_dict)
        self.node_post_process(split_layer)

        squeeze_params = np.array(
            [self.get_value_by_key_or_index("dim", 1, 0)], dtype=np.int64
        )
        if len(self._output_shape) == 1:
            squeeze_layer = SqueezeFunc(self._source_node, auto_gen=False)
            squeeze_layer.add_bottom_top(in_names=[split_out_names[0]])
            squeeze_layer.generate_node(self._name + "_squeeze", squeeze_params)
            self.node_post_process(squeeze_layer)
        else:
            for idx in range(len(self._output_shape)):
                squeeze_layer = SqueezeFunc(self._source_node, auto_gen=False)
                squeeze_layer.add_bottom_top(
                    in_names=[split_out_names[idx]],
                    out_names=[self._name + "_" + str(idx)],
                )
                squeeze_layer.generate_node(
                    self._name + "_squeeze_" + str(idx), squeeze_params
                )
                self.node_post_process(squeeze_layer)
