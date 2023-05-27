import torch
import copy
from torch.fx import GraphModule


class BrocolliGraphModule(GraphModule):
    def __init__(self, root, graph, class_name="GraphModule"):
        graph_copy = copy.deepcopy(graph)
        super().__init__(root, graph_copy, class_name)
        self.class_name = class_name

    def __deepcopy__(self, memo):
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        graph_module = BrocolliGraphModule(fake_mod, self.graph, self.class_name)
        for key, value in fake_mod.__dict__.items():
            if (
                key not in graph_module.__dict__.keys()
            ):  # to handle those user-defined attr
                graph_module.__dict__[key] = value

        return graph_module

    def dump_json(self, filename=None):
        import json

        json_dict = {}
        self.graph.print_tabular()
        for node in list(self.graph.nodes):
            if node.op == "call_module":
                module = dict(self.named_modules())[node.target]
                if hasattr(module, "output_scale"):
                    output_scale = float(module.output_scale)
                    json_dict[node.name] = output_scale

        json_object = json.dumps(json_dict, indent=4)
        with open(f"{filename}", "w") as outfile:
            outfile.write(json_object)
