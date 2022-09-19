import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import Node
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval

from .utils import replace_node_module
from .pattern import register_fusion_pattern


class MatchAllNode:
    """A node pattern that matches all nodes"""

    pass


def is_match(modules, node, pattern, max_uses=sys.maxsize):
    """Matches a node in fx against a pattern"""
    if isinstance(pattern, tuple):
        self_match, *arg_matches = pattern
        if self_match is getattr:
            assert len(pattern) == 2, "Expecting getattr pattern to have two elements"
            arg_matches = []
    else:
        self_match = pattern
        arg_matches = []

    if isinstance(self_match, type) and issubclass(self_match, MatchAllNode):
        return True

    if len(node.users) > max_uses:
        return False

    if isinstance(self_match, type) and issubclass(self_match, torch.nn.Module):
        if node.op != "call_module":
            return False
        if not type(modules[node.target]) == self_match:
            return False
    elif callable(self_match):
        if node.op != "call_function" or node.target is not self_match:
            return False
        elif node.target is getattr:
            if node.args[1] != pattern[1]:
                return False
    elif isinstance(self_match, str):
        if node.op != "call_method" or node.target != self_match:
            return False
    elif node.target != self_match:
        return False

    if not arg_matches:
        return True

    if len(arg_matches) != len(node.args):
        return False

    return all(
        is_match(modules, node, arg_match, max_uses=1)
        for node, arg_match in zip(node.args, arg_matches)
    )


@register_fusion_pattern((torch.nn.BatchNorm1d, torch.nn.Conv1d))
@register_fusion_pattern((torch.nn.BatchNorm2d, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.BatchNorm3d, torch.nn.Conv3d))
class ConvBNFusion:
    def __init__(self, quantizer, node):
        self.bn_node = None
        if isinstance(
            quantizer.modules[node.target],
            (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d),
        ):
            self.bn_node = node
            self.bn = quantizer.modules[self.bn_node.target]
            assert isinstance(node.args[0], Node)
            node = node.args[0]
        assert node.op == "call_module"
        self.conv_node = node
        self.conv = quantizer.modules[self.conv_node.target]

    def fuse(self, fused_graph, modules):
        op_list = []
        assert self.bn_node is not None
        op_list.append(self.conv)
        op_list.append(self.bn)

        fused_conv = fuse_conv_bn_eval(self.conv, self.bn)
        replace_node_module(self.conv_node, modules, fused_conv)

        if self.bn_node is not None:
            replace_node_module(self.bn_node, modules, torch.nn.Identity())

        self.bn_node.replace_all_uses_with(self.conv_node)
        fused_graph.erase_node(self.bn_node)
