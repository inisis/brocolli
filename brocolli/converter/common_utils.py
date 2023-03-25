import re
import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval


def get_shape(obj):
    return obj["shape"]


def get_dtype(obj):
    return obj["dtype"]


def map_reduce(args, fn):
    shape_list = []
    if isinstance(args, tuple):
        shape = sum(list(map_reduce(elem, fn) for elem in args), [])
    elif isinstance(args, list):
        shape = sum(list(map_reduce(elem, fn) for elem in args), [])
    elif args == None:
        shape = []
    else:
        shape = [fn(args)]

    shape_list.extend(shape)

    return shape_list


def get_torch_size(obj):
    return torch.Size(obj)


def gen_torch_tensor(obj):
    return torch.rand(obj).to(torch.int32)


def gen_numpy_data(obj):
    return obj.numpy()


def map_replace(args, fn):
    if all(isinstance(element, int) for element in args):
        return fn(args)
    elif isinstance(args, tuple):
        return list(map_replace(elem, fn) for elem in args)
    elif isinstance(args, list):
        return list(map_replace(elem, fn) for elem in args)
    else:
        return fn(args)


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


def _parent_name(target):
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit(".", 1)
    return parent[0] if parent else "", name


def replace_node_module(node, modules, new_module):
    assert isinstance(node.target, str)
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, new_module)
