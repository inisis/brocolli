from loguru import logger
import numpy as np
from onnx import helper
from onnx import TensorProto as tp


from .base_layer import BaseLayer


class SumFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(SumFunc, self).__init__(source_node, module, auto_gen)

    def get_sum_attr(self):
        attr_dict = {"keepdims": 1}

        attr_dict["keepdims"] = self.get_value_by_key_or_index("keepdims", 2, False)

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):
        axes = self._source_node.args[1]
        if isinstance(axes, int):
            axes = [axes]

        params = np.array(axes)

        self.create_params(self._name + "_sum", params)
        attr_dict = self.get_sum_attr()
        node = helper.make_node(
            "ReduceSum", self._in_names, self._out_names, self._name, **attr_dict
        )

        logger.info("sum_layer: " + self._name + " created")
        self._node.append(node)
