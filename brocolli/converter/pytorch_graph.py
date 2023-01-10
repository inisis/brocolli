import torch
import torch.nn as nn

import torch.fx
from torch.fx import Tracer, Interpreter
from torch.fx.graph_module import GraphModule
from torch.fx.node import map_aggregate

from .utils import get_function_name, map_replace, gen_torch_tensor
from .pytorch_layer.transformer import (
    Transformer,
    TransformerEncoder,
    TransformerDecoder,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)
from .pytorch_layer.mha import MultiheadAttention

from .pytorch_layer.layernorm import LayerNorm


class BrocolliTracer(Tracer):
    def __init__(self, *args, customed_leaf_module=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.customed_leaf_module = customed_leaf_module

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str):
        if self.customed_leaf_module and isinstance(m, self.customed_leaf_module):

            return True

        if hasattr(m, "_is_leaf_module") and m._is_leaf_module:
            return True

        return m.__module__.startswith("torch.nn") and not isinstance(
            m, torch.nn.Sequential
        )


class BrocolliShapeRunner(Interpreter):
    def __init__(self, module, dynamic_batch=False):
        super(BrocolliShapeRunner, self).__init__(module)
        self.dynamic_batch = dynamic_batch

    def run_node(self, n):
        result = super().run_node(n)

        found_tensor = False

        def extract_tensor_metadata(result: torch.Tensor):
            meta_info = {}
            shape = list(result.shape)
            if self.dynamic_batch:
                shape[0] = -1
            meta_info["shape"] = torch.Size(shape)
            meta_info["dtype"] = result.dtype

            return meta_info

        def extract_tensor_meta(obj):
            if isinstance(obj, torch.Tensor):
                nonlocal found_tensor
                found_tensor = True
                return extract_tensor_metadata(obj)
            else:
                return obj

        meta = map_aggregate(result, extract_tensor_meta)
        if found_tensor:
            n.meta["tensor_meta"] = meta

        n.meta["type"] = type(result)
        return result


class PytorchGraph:
    def __init__(self, model, inputs, concrete_args=None, dynamic_batch=False):
        super(PytorchGraph, self).__init__()
        self.model = model
        self.inputs = inputs
        self.concrete_args = concrete_args
        self.dynamic_batch = dynamic_batch

        if isinstance(self.model, GraphModule):
            self.graph_module = self.model
            self.shape_inference()
        elif isinstance(self.model, nn.Module):
            self.replace(self.model)
            self.tracer = BrocolliTracer()
            self.graph = self.tracer.trace(self.model, concrete_args)
            self.graph_module = GraphModule(self.tracer.root, self.graph)
            if concrete_args is not None:
                self.trace_prune(self.graph_module)
            self.shape_inference()
        else:
            raise Exception(
                "model must be a torch.nn.Module \
                            or a torch.fx.GraphModule"
            )

        self.graph = self.graph_module.graph
        self.nodes = list(self.graph_module.graph.nodes)
        self.graph.print_tabular()

    def replace(self, model):
        for name, module in model.named_children():
            if isinstance(module, nn.Transformer):
                converted_module = Transformer.from_torch(module)
                setattr(model, name, converted_module)
            elif isinstance(module, nn.TransformerEncoder):
                converted_module = TransformerEncoder.from_torch(module)
                setattr(model, name, converted_module)
            elif isinstance(module, nn.TransformerDecoder):
                converted_module = TransformerDecoder.from_torch(module)
                setattr(model, name, converted_module)
            elif isinstance(module, nn.TransformerEncoderLayer):
                converted_module = TransformerEncoderLayer.from_torch(module)
                setattr(model, name, converted_module)
            elif isinstance(module, nn.TransformerDecoderLayer):
                converted_module = TransformerDecoderLayer.from_torch(module)
                setattr(model, name, converted_module)
            elif isinstance(module, nn.MultiheadAttention):
                converted_module = MultiheadAttention.from_torch(module)
                setattr(model, name, converted_module)
            elif list(module.named_children()):
                self.replace(module)

    def placeholder_prune(self, graph_module):
        for node in list(graph_module.graph.nodes):
            if node.op == "placeholder" and node.next.op == "call_function":
                function_name = get_function_name(node.next.target)
                if function_name == "eq" and node.next.next.op == "call_function":
                    function_name = get_function_name(node.next.next.target)
                    if function_name == "_assert":
                        graph_module.graph.erase_node(node.next.next)
                        graph_module.graph.erase_node(node.next)
                        graph_module.graph.erase_node(node)

    def trace_prune(self, graph_module):
        self.placeholder_prune(graph_module)

    def shape_inference(self):
        shape_runner = BrocolliShapeRunner(self.graph_module, self.dynamic_batch)
        if self.concrete_args is not None:
            shape_runner.run(*self.inputs + tuple(self.concrete_args.values()))
        else:
            shape_runner.run(*self.inputs)
