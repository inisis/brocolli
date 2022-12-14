import torch
import torch.nn.functional as F
from loguru import logger
from onnx import helper
from onnxruntime_extensions import onnx_op, PyOp

from .base_layer import BaseLayer


class GELULayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(GELULayer, self).__init__(source_node, module, auto_gen)

    def generate_node(self, name=None, params=None, attr_dict=None):

        node = helper.make_node(
            "GELU",
            self._in_names,
            self._out_names,
            self._name,
            domain="ai.onnx.contrib",
        )
        logger.info("gelu_layer: " + self._name + " created")
        self._node.append(node)


@onnx_op(
    op_type="GELU",
    inputs=[PyOp.dt_float],
    outputs=[PyOp.dt_float],
)
def GELU(x):
    x = torch.from_numpy(x)
    output = F.gelu(x)

    return output
