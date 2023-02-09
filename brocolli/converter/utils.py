import re
import contextlib
import torch
import torch.nn as nn
import numpy as np
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


def graph_constant_fold_inplace(graph):
    for node in graph.nodes:
        if node.op == "Identity" or node.op == "Dropout":
            inp_node = node.i()
            inp_node.outputs = node.outputs
            node.outputs.clear()


def find_gelu_nodes(graph):
    # fmt: off
    '''
             x
         /      \
         |     Div
         |      |
         |     Erf
         |      |
         |     Add
         \      /
            Mul
             |
            Mul
    '''
    # fmt: on
    out_nodes = []
    for node in graph.nodes:
        with contextlib.suppress(IndexError):
            if node.op == "Mul":
                if (
                    node.i(0).op == "Mul"
                    and node.i(0).i(1).op == "Add"
                    and node.i(0).i(1).i(0).op == "Erf"
                    and node.i(0).i(1).i(0).i(0).op == "Div"
                ):
                    input_variable = node.i(0).i(1).i(0).i(0).inputs[0]
                    mul_node = node.i(0)
                    div_node = node.i(0).i(1).i(0).i(0)

                    input_variable.outputs.remove(mul_node)
                    input_variable.outputs.remove(div_node)

                    output_variable = node.outputs[0]
                    output_variable.inputs.clear()
                    out_nodes += [
                        {
                            "inps": [input_variable],
                            "outs": [output_variable],
                        }
                    ]

    return out_nodes


def find_swish_nodes(graph):
    # fmt: off
    '''
             x
         /      \
         |    Sigmoid
         \      /
            Mul
    '''
    # fmt: on
    out_nodes = []
    for node in graph.nodes:
        with contextlib.suppress(IndexError):
            if node.op == "Mul":
                if node.i(1).op == "Sigmoid":
                    input_variable = node.i(1).inputs[0]
                    mul_node = node.i(1)
                    sigmoid_node = node

                    input_variable.outputs.remove(mul_node)
                    input_variable.outputs.remove(sigmoid_node)

                    output_variable = node.outputs[0]
                    output_variable.inputs.clear()
                    out_nodes += [
                        {
                            "inps": [input_variable],
                            "outs": [output_variable],
                        }
                    ]

    return out_nodes


def find_layernorm_nodes(graph):
    # fmt: off
    '''
             x
         /      \
         |  ReduceMean
         \      /
            Sub
         /      \
         |     Pow
         |      |
         |  ReduceMean
         |      |
         |     Add
         |      |
         |     Sqrt
         \      /
            Div
             |
            Mul
             |
            Add
    '''
    # fmt: on
    out_nodes = []
    for node in graph.nodes:
        with contextlib.suppress(IndexError):
            if node.op == "Add":
                if (
                    node.i(0).op == "Mul"
                    and node.i(0).i(0).op == "Div"
                    and node.i(0).i(0).i(0).op == "Sub"
                    and node.i(0).i(0).i(1).op == "Sqrt"
                    and node.i(0).i(0).i(1).i(0).op == "Add"
                    and node.i(0).i(0).i(1).i(0).i(0).op == "ReduceMean"
                    and node.i(0).i(0).i(1).i(0).i(0).i(0).op == "Pow"
                    and node.i(0).i(0).i(1).i(0).i(0).i(0).i(0).op == "Sub"
                    and node.i(0).i(0).i(1).i(0).i(0).i(0).i(0).i(1).op == "ReduceMean"
                ):
                    input_variable = (
                        node.i(0).i(0).i(1).i(0).i(0).i(0).i(0).i(1).inputs[0]
                    )
                    sub_node = node.i(0).i(0).i(1).i(0).i(0).i(0).i(0).i(1)
                    reducemean_node = node.i(0).i(0).i(1).i(0).i(0).i(0).i(0)

                    input_variable.outputs.remove(sub_node)
                    input_variable.outputs.remove(reducemean_node)

                    weight_variable = node.i(0).inputs[1]
                    bias_variable = node.inputs[1]

                    output_variable = node.outputs[0]
                    output_variable.inputs.clear()
                    out_nodes += [
                        {
                            "inps": [
                                input_variable,
                                weight_variable,
                                bias_variable,
                            ],
                            "outs": [output_variable],
                            "attrs": {
                                "attrs": str(
                                    {
                                        "axis": node.i(0)
                                        .i(0)
                                        .i(1)
                                        .i(0)
                                        .i(0)
                                        .attrs["axes"][0],
                                        "eps": float(
                                            node.i(0).i(0).i(1).i(0).inputs[1].values
                                        ),
                                    }
                                )
                            },
                        }
                    ]
    return out_nodes


@gs.Graph.register()
def replace_gelu(self, inputs, outputs, name):
    return self.layer(
        op="GELU", inputs=inputs, outputs=outputs, name=name, domain="ai.onnx.contrib"
    )


@gs.Graph.register()
def replace_swish(self, inputs, outputs, name):
    return self.layer(
        op="Swish", inputs=inputs, outputs=outputs, name=name, domain="ai.onnx.contrib"
    )


@gs.Graph.register()
def replace_layernorm(self, inputs, outputs, attrs, name):
    return self.layer(
        op="LayerNormalization",
        inputs=inputs,
        outputs=outputs,
        name=name,
        attrs=attrs,
        domain="ai.onnx.contrib",
    )


def optimize_model(model):
    graph = gs.import_onnx(model)
    graph.fold_constants().cleanup()
    gelu_nodes = find_gelu_nodes(graph)
    swish_node = find_swish_nodes(graph)
    layernorm_node = find_layernorm_nodes(graph)

    for i, itn in enumerate(gelu_nodes):
        inputs = itn["inps"]
        outputs = itn["outs"]
        name = "gelu_{}".format(i)
        graph.replace_gelu(inputs, outputs, name)

    for i, itn in enumerate(swish_node):
        inputs = itn["inps"]
        outputs = itn["outs"]
        name = "swish_{}".format(i)
        graph.replace_swish(inputs, outputs, name)

    for i, itn in enumerate(layernorm_node):
        inputs = itn["inps"]
        outputs = itn["outs"]
        attrs = itn["attrs"]
        name = "layernorm_{}".format(i)
        graph.replace_layernorm(inputs, outputs, attrs, name)

    graph_constant_fold_inplace(graph)
    graph.cleanup(remove_unused_graph_inputs=True).toposort()
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

numpy_to_torch_dtype_dict = {
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}

torch_to_numpy_dtype_dict = {
    value: key for (key, value) in numpy_to_torch_dtype_dict.items()
}


def pytorch_dtype_to_onnx(scalar_type):
    torch_type = scalar_type_to_pytorch_type.index(scalar_type)
    onnx_type = scalar_type_to_onnx[torch_type]
    return onnx_type


def numpy_dtype_to_torch(scalar_type):
    return numpy_to_torch_dtype_dict[scalar_type]


def torch_dtype_to_numpy(scalar_type):
    return torch_to_numpy_dtype_dict[scalar_type]
