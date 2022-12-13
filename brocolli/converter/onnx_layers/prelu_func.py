from loguru import logger
import numpy as np
from onnx import helper
from onnx import TensorProto as tp


from .base_layer import BaseLayer


class PReluFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(PReluFunc, self).__init__(source_node, module, auto_gen)

    def add_bottom_top(self, in_names=None, out_names=None):
        if in_names is None:
            in_names = [self.recursive_find_name(self._source_node.args[0])]

        if out_names is None:
            out_names = [self._name]

        self._in_names.extend(in_names)
        self._out_names.extend(out_names)

    def generate_node(self, name=None, params=None, attr_dict=None):

        target_atoms = self._source_node.args[1].target.split(".")
        attr_itr = self._module
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(
                    f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
                )
            attr_itr = getattr(attr_itr, atom)
            params = attr_itr.detach().numpy()
            shape = self._output_shape[0]
            param_shape = [1] * len(shape)
            param_shape[1] = params.shape[0]
            params = params.reshape(param_shape)
            self.create_params(self._name + "_prelu", params)

        node = helper.make_node("PRelu", self._in_names, self._out_names, self._name)

        logger.info("prelu_layer: " + self._name + " created")
        self._node.append(node)
