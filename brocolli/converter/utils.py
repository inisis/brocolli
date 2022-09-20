import re

import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval


def fuse_all_conv_bn(model):
    stack = []
    for name, module in model.named_children():
        if list(module.named_children()):
            fuse_all_conv_bn(module)

        if isinstance(module, nn.BatchNorm2d):
            if not stack:
                continue
            if isinstance(stack[-1][1], nn.Conv2d):
                setattr(model, stack[-1][0], fuse_conv_bn_eval(stack[-1][1], module))
                setattr(model, name, nn.Identity())
        elif isinstance(module, nn.BatchNorm1d):
            if not stack:
                continue
            if isinstance(stack[-1][1], nn.Linear):
                setattr(model, stack[-1][0], fuse_linear_bn_eval(stack[-1][1], module))
                setattr(model, name, nn.Identity())
        else:
            stack.append((name, module))


def get_function_name(node_target):
    function_name = re.findall(
        r"(?:function|method) ([a-z|_|0-9]+.*?)", str(node_target)
    )[0]

    return function_name
