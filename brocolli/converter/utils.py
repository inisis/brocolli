import re
import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval
import onnx_graphsurgeon as gs


def get_shape(obj):
    return obj["shape"]


def map_reduce(args, fn):
    shape_list = []
    if isinstance(args, tuple):
        shape = sum(list(map_reduce(elem, fn) for elem in args), [])
    elif isinstance(args, list):
        shape = sum(list(map_reduce(elem, fn) for elem in args), [])
    else:
        shape = [fn(args)]

    shape_list.extend(shape)

    return shape_list


def get_torch_size(obj):
    return torch.Size(obj)


def gen_torch_tensor(obj):
    return torch.rand(obj)


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


def graph_cleanup_inplace(graph):
    graph.cleanup()


def graph_constant_fold_inplace(graph):
    for node in graph.nodes:
        if node.op == "Identity" or node.op == "Dropout":
            inp_node = node.i()
            inp_node.outputs = node.outputs
            node.outputs.clear()


def optimize_model(model):
    graph = gs.import_onnx(model)
    graph_constant_fold_inplace(graph)
    graph_cleanup_inplace(graph)
    model = gs.export_onnx(graph)

    return model
