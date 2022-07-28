import torch
import torch.nn as nn

import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.graph_module import GraphModule

from .utils import get_function_name

class PytorchGraph():

    def __init__(self, model, input_shape, concrete_args=None):
        super(PytorchGraph, self).__init__()  
        self.model = model
        self.input_shape = input_shape
        self.concrete_args = concrete_args

        if isinstance(self.model, GraphModule):
            self.trace = self.model
            self.shape_inference()
        elif isinstance(self.model, nn.Module):
            self.trace = torch.fx.symbolic_trace(self.model, concrete_args)
            if concrete_args is not None:
                self.trace_prune(self.trace)
            self.shape_inference()
        else:
            raise Exception("model must be a torch.nn.Module or a torch.fx.GraphModule")

        self.graph = self.trace.graph
        self.nodes = list(self.trace.graph.nodes)

    def placeholder_prune(self, trace):
        for node in list(trace.graph.nodes):
            if node.op == 'placeholder' and node.next.op == 'call_function':
                function_name = get_function_name(node.next.target)
                if function_name == 'eq' and node.next.next.op == 'call_function':
                    function_name = get_function_name(node.next.next.target)
                    if function_name == '_assert':
                        trace.graph.erase_node(node.next.next)
                        trace.graph.erase_node(node.next)
                        trace.graph.erase_node(node)

    def trace_prune(self, trace):
        self.placeholder_prune(trace)

    def gen_input_tensor(self, shapes):
        input_tensor = []
        for shape in shapes:
            if isinstance(shape, (tuple, list)):
                if all(isinstance(element, int) for element in shape):
                    input_tensor.append(torch.rand(shape).to(torch.float32))
                else:
                    input_tensor.append(self.gen_input_tensor(shape))
            else:
                input_tensor.append(torch.rand(shape).to(torch.float32))

        return input_tensor   

    def shape_inference(self):  
        dummy_input = self.gen_input_tensor(self.input_shape)
        ShapeProp(self.trace).propagate(*dummy_input)