import torch
import torch.nn.functional as F
from onnxruntime_extensions import onnx_op, PyOp

from .base_layer import BaseLayer
from .sigmoid_layer import SigmoidLayer
from .mul_layer import MulLayer


class SwishLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(SwishLayer, self).__init__(source_node, module, auto_gen)

    def add_bottom_top(self, in_names=None, out_names=None):
        pass

    def generate_node(self, name=None, params=None, attr_dict=None):
        sigmoid_layer = SigmoidLayer(self._source_node, self._module, auto_gen=False)
        sigmoid_layer.add_bottom_top(out_names=[self._source_node.name + "_sigmoid"])
        sigmoid_layer.generate_node(self._source_node.name + "_sigmoid")
        self.node_post_process(sigmoid_layer)

        mul_layer = MulLayer(self._source_node, self._module, auto_gen=False)
        mul_layer.add_bottom_top(
            in_names=[sigmoid_layer._in_names[0], self._source_node.name + "_sigmoid"]
        )
        mul_layer.generate_node(self._source_node.name)
        self.node_post_process(mul_layer)


@onnx_op(
    op_type="Swish",
    inputs=[PyOp.dt_float],
    outputs=[PyOp.dt_float],
)
def Swish(x):
    x = torch.from_numpy(x)
    output = F.silu(x)

    return output
