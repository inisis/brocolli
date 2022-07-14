import torch
import torch.nn as nn

import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.graph_module import GraphModule

class PytorchGraph():

    def __init__(self, model, input_shape):
        super(PytorchGraph, self).__init__()  
        self.model = model
        self.input_shape = input_shape

        if isinstance(self.model, GraphModule):
            self.trace = self.model
            self.shape_inference()
        elif isinstance(self.model, nn.Module):
            self.trace = torch.fx.symbolic_trace(self.model)
            self.shape_inference()
        else:
            raise Exception("model must be a torch.nn.Module or a torch.fx.GraphModule")

        self.graph = self.trace.graph
        self.nodes = list(self.trace.graph.nodes)

    def shape_inference(self):
        if isinstance(self.input_shape, tuple):
            dummy_input = []
            names = []
            for idx, each in enumerate(self.input_shape):
                dummy = torch.ones(each)
                dummy_input.append(dummy)
                names.append("data_" + str(idx))
            dummy_input = tuple(dummy_input)
            ShapeProp(self.trace).propagate(*dummy_input)
        else:
            dummy_input = torch.ones(self.input_shape)
            ShapeProp(self.trace).propagate(dummy_input)      