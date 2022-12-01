from loguru import logger
import numpy as np
from onnx import helper


from .base_layer import BaseLayer


class ReductionLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(ReductionLayer, self).__init__(source_node, module, auto_gen)

    def generate_node(self, name=None, params=None, attr_dict=None):
        axis = self._layer.reduction_param.axis
        if axis == len(shape):
            axes = [axis]
        else:
            axes = np.arange(axis, len(shape)).tolist()

        if self._layer.reduction_param.operation == 1:
            node = helper.make_node(
                "ReduceSum",
                self._in_names,
                self._out_names,
                self._name,
                keepdims=0,
                axes=axes,
            )
        elif self._layer.reduction_param.operation == 2:
            node = helper.make_node(
                "ReduceSum",
                self._in_names,
                self._out_names,
                self._name,
                keepdims=0,
                axes=axes,
            )
        elif self._layer.reduction_param.operation == 3:
            node = helper.make_node(
                "ReduceSumSquare",
                self._in_names,
                self._out_names,
                self._name,
                keepdims=0,
                axes=axes,
            )
        elif self._layer.reduction_param.operation == 4:
            node = helper.make_node(
                "ReduceMean",
                self._in_names,
                self._out_names,
                self._name,
                keepdims=0,
                axes=axes,
            )

        logging.info("eltwise_layer: " + self._name + " created")
        self._node.append(node)
