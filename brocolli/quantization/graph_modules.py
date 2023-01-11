import torch
import copy
from torch.fx import GraphModule


class BrocolliGraphModule(GraphModule):
    def __init__(self, root, graph, class_name="GraphModule"):
        super().__init__(root, graph, class_name)
        self.class_name = class_name

    def __deepcopy__(self, memo):
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        graph_copy = copy.deepcopy(self.graph)
        graph_module = BrocolliGraphModule(fake_mod, graph_copy, self.class_name)
        graph_module.__dict__.update(fake_mod.__dict__)

        return graph_module
