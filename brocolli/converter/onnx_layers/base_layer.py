from onnx import helper
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from torch.fx.node import Node
from ..utils import (
    get_function_name,
    get_shape,
    map_reduce,
    get_dtype,
    pytorch_dtype_to_onnx,
)


class BaseLayer(object):
    def __init__(self, source_node, module=None, auto_gen=True):
        self._source_node = source_node
        self._module = module
        self._auto_gen = auto_gen
        self._in_tensor_value_info = []
        self._init_tensor = []
        self._out_tensor_value_info = []
        self._value_info = []
        self._node = []
        self._name = self._source_node.name
        self._input_shape = []
        self._input_dtype = []
        if len(self._source_node.all_input_nodes) != 0:
            for node in self._source_node.all_input_nodes:
                if "tensor_meta" in list(node.meta.keys()):
                    self._input_shape.extend(
                        map_reduce(node.meta["tensor_meta"], get_shape)
                    )
                    self._input_dtype.extend(
                        map_reduce(node.meta["tensor_meta"], get_dtype)
                    )

        self._output_type = self._source_node.meta["type"]
        self._output_shape = []
        self._output_dtype = []
        if "tensor_meta" in list(self._source_node.meta.keys()):
            self._output_shape.extend(
                map_reduce(self._source_node.meta["tensor_meta"], get_shape)
            )
            self._output_dtype.extend(
                map_reduce(self._source_node.meta["tensor_meta"], get_dtype)
            )

        self._in_names = []
        self._out_names = []

        if self._auto_gen:
            self.add_bottom_top()
            self.generate_node()

    def create_params(self, param_name, param, param_type=None):
        if param is None:
            self._in_names.append("")
        else:
            if param_type is None:
                param_type = NP_TYPE_TO_TENSOR_TYPE[param.dtype]
            param_shape = param.shape
            param_tensor_value_info = helper.make_tensor_value_info(
                param_name, param_type, param_shape
            )
            param_tensor = helper.make_tensor(
                param_name, param_type, param_shape, param.flatten()
            )
            self._in_names.append(param_name)
            self._in_tensor_value_info.append(param_tensor_value_info)
            self._init_tensor.append(param_tensor)

    def generate_node(self, name=None, params=None, attr_dict=None):
        pass

    def get_value_by_key_or_index(self, key, index, default=None):
        if key in self._source_node.kwargs:
            return self._source_node.kwargs[key]
        elif index < len(self._source_node.args):
            return self._source_node.args[index]
        else:
            return default

    def recursive_find_name(self, node):
        if node.op == "placeholder":
            return node.name
        elif node.op == "output":
            return node.name
        elif node.op == "call_module":
            return node.name
        elif node.op == "call_function":
            function_name = get_function_name(node.target)
            if function_name == "getitem":
                if isinstance(node.args[1], int):
                    node_name = node.args[0].name + "_" + str(node.args[1])
                    return node_name
                else:
                    return node.name
            else:
                return node.name
        elif node.op == "call_method":
            if str(node.target) == "contiguous":
                node_ = node.args[0]
                return self.recursive_find_name(node_)
            else:
                return node.name
        elif node.op == "get_attr":
            return node.name

    def add_bottom_top(self, in_names=None, out_names=None):
        if in_names is None:
            for node in self._source_node.args:
                if isinstance(node, Node):
                    bottom_name = self.recursive_find_name(node)
                    if bottom_name is None:
                        continue
                    self._in_names.append(bottom_name)
                elif isinstance(node, list) or isinstance(
                    node, tuple
                ):  # cat function args[0]
                    for node_ in node:
                        if isinstance(node_, Node):
                            bottom_name = self.recursive_find_name(node_)
                            if bottom_name is None:
                                continue
                            self._in_names.append(bottom_name)
                else:
                    continue
        else:
            if not isinstance(in_names, list):
                raise Exception("custom in_names must be list")

            self._in_names.extend(in_names)

        if out_names is None:
            self._out_names.append(self._name)
        else:
            if not isinstance(out_names, list):
                raise Exception("custom out_names must be list")

            self._out_names.extend(out_names)

        if len(self._output_shape) == len(self._out_names):
            onnx_type = pytorch_dtype_to_onnx(self._output_dtype[0])
            param_tensor_value_info = helper.make_tensor_value_info(
                self._out_names[0], onnx_type, self._output_shape[0]
            )
            self._value_info.append(param_tensor_value_info)

    def node_post_process(self, onnx_layer):
        if onnx_layer._node:
            self._node.extend(onnx_layer._node)
        self._in_tensor_value_info.extend(onnx_layer._in_tensor_value_info)
        self._out_tensor_value_info.extend(onnx_layer._out_tensor_value_info)
        self._init_tensor.extend(onnx_layer._init_tensor)
