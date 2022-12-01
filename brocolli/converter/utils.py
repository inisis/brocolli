import re
import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval
import onnx_graphsurgeon as gs
from onnx import TensorProto as tp


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


scalar_type_to_pytorch_type = [
    torch.uint8,  # 0
    torch.int8,  # 1
    torch.short,  # 2
    torch.int,  # 3
    torch.int64,  # 4
    torch.half,  # 5
    torch.float,  # 6
    torch.double,  # 7
    torch.complex32,  # 8
    torch.complex64,  # 9
    torch.complex128,  # 10
    torch.bool,  # 11
]

cast_pytorch_to_onnx = {
    "Byte": tp.UINT8,
    "Char": tp.INT8,
    "Double": tp.DOUBLE,
    "Float": tp.FLOAT,
    "Half": tp.FLOAT16,
    "Int": tp.INT32,
    "Long": tp.INT64,
    "Short": tp.INT16,
    "Bool": tp.BOOL,
    "ComplexFloat": tp.COMPLEX64,
    "ComplexDouble": tp.COMPLEX128,
    "Undefined": tp.UNDEFINED,
}

scalar_type_to_onnx = [
    cast_pytorch_to_onnx["Byte"],
    cast_pytorch_to_onnx["Char"],
    cast_pytorch_to_onnx["Short"],
    cast_pytorch_to_onnx["Int"],
    cast_pytorch_to_onnx["Long"],
    cast_pytorch_to_onnx["Half"],
    cast_pytorch_to_onnx["Float"],
    cast_pytorch_to_onnx["Double"],
    cast_pytorch_to_onnx["Undefined"],
    cast_pytorch_to_onnx["ComplexFloat"],
    cast_pytorch_to_onnx["ComplexDouble"],
    cast_pytorch_to_onnx["Bool"],
]
