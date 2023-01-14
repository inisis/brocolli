import torch
from loguru import logger
from onnx import helper
from onnxruntime_extensions import onnx_op, PyOp

from .base_layer import BaseLayer
import torch.nn.functional as F
import math


class CustomLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(CustomLayer, self).__init__(source_node, module, auto_gen)

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
        node = helper.make_node(
            self._module.__class__.__name__,
            self._in_names,
            self._out_names,
            self._name,
            domain="ai.onnx.contrib",
        )
        logger.info(
            self._module.__class__.__name__ + "_layer: " + self._name + " created"
        )
        self._node.append(node)


@onnx_op(
    op_type="PositionEmbeddingSine",
    inputs=[PyOp.dt_float],
    outputs=[PyOp.dt_float, PyOp.dt_float],
)
def PositionEmbeddingSine(mask):
    mask = torch.from_numpy(mask)
    num_pos_feats = 128
    temperature = 10000
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    mask = F.interpolate(mask.unsqueeze(0), size=(28, 38)).squeeze(0)
    mask1 = mask.to(torch.bool)
    assert mask1 is not None
    not_mask = ~mask1
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)
    if True:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi
    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    pos_y = torch.stack(
        (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).to(torch.float32)

    return pos, mask
