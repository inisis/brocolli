import torch
from loguru import logger
from onnx import helper
from onnxruntime_extensions import onnx_op, PyOp

from .base_layer import BaseLayer


class LayerNormLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(LayerNormLayer, self).__init__(source_node, module, auto_gen)

    def get_layernorm_attr(self):
        attr_dict = {"axis": "1", "eps": "1e-5"}

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):

        self.create_params(self._name + "_weight", self._module.weight)
        self.create_params(self._name + "_bias", self._module.bias)
        attr_dict = self.get_layernorm_attr()
        node = helper.make_node(
            "LayerNormalization",
            self._in_names,
            self._out_names,
            self._name,
            domain="ai.onnx.contrib",
            **attr_dict
        )
        logger.info("layernorm_layer: " + self._name + " created")
        self._node.append(node)


@onnx_op(
    op_type="LayerNormalization",
    inputs=[PyOp.dt_float, PyOp.dt_float, PyOp.dt_float],
    outputs=[PyOp.dt_float],
    attrs=["axis", "eps"],
)
def LayerNormalization(x, weight, bias, **kwargs):
    eps = float(kwargs.get("eps", 1e-5))
    axis = int(kwargs.get("axis", -1))

    x = torch.from_numpy(x)
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    std = (var + eps).sqrt()
    y = (x - mean) / std
    y *= weight
    y += bias

    return y
