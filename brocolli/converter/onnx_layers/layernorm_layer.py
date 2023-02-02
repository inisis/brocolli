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
        attr_dict["axis"] = str(-len(self._module.normalized_shape))
        attr_dict["eps"] = str(self._module.eps)

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):

        self.create_params(self._name + "_weight", self._module.weight.detach().numpy())
        self.create_params(self._name + "_bias", self._module.bias.detach().numpy())
        attr_dict = self.get_layernorm_attr()
        node = helper.make_node(
            "LayerNormalization",
            self._in_names,
            self._out_names,
            self._name,
            domain="ai.onnx.contrib",
            attrs=str(attr_dict),
        )
        logger.info("layernorm_layer: " + self._name + " created")
        self._node.append(node)


@onnx_op(
    op_type="LayerNormalization",
    inputs=[PyOp.dt_float, PyOp.dt_float, PyOp.dt_float],
    outputs=[PyOp.dt_float],
    attrs=["attrs"],
)
def LayerNormalization(x, weight, bias, **kwargs):
    attrs = eval(kwargs["attrs"])
    eps = float(attrs.get("eps", 1e-5))
    axis = int(attrs.get("axis", -1))
    x = torch.from_numpy(x)
    dim = list(range(axis, 0))
    mean = x.mean(dim=dim, keepdim=True)
    x -= mean
    var = (x**2).mean(dim=dim, keepdim=True)
    std = (var + eps).sqrt()
    y = x / std
    y *= weight
    y += bias

    return y
