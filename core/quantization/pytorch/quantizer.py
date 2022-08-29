import copy
from loguru import logger
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.fx
from torch.fx import Tracer
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.graph_module import GraphModule

from .profiler import FXProfiler
from .qconfig import get_qconfig

from .quantization_layers.input import Input
from .quantization_layers.output import Output
from .quantization_layers.conv import Conv2d
from .quantization_layers.relu import ReLU
from .quantization_layers.pooling import MaxPool2d
from .quantization_layers.linear import Linear

from torchvision import datasets, transforms


def activation_pre_hook(self, input):
    if hasattr(self, "activation_pre_process"):
        self.activation_pre_process(input[0])


def activation_post_hook(self, input, output):
    if hasattr(self, "activation_post_process"):
        self.activation_post_process(output[0])


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


class PytorchQuantizer:
    def __init__(self, model, input_shape, concrete_args=None):
        super(PytorchQuantizer, self).__init__()
        self.model = model
        self.input_shape = input_shape
        if isinstance(input_shape, (tuple, list)) and all(
            isinstance(element, int) for element in input_shape
        ):
            self.input_shape = [input_shape]
        self.concrete_args = concrete_args
        self.qconfig = None
        self.qconfig_dict = {"": self.qconfig}

        self.graph_module = self.get_graph_module(self.model, self.concrete_args)

    def get_graph_module(self, model, concrete_args):
        if isinstance(model, GraphModule):
            trace = model
        elif isinstance(model, nn.Module):
            tracer = BrocolliTracer()
            graph = tracer.trace(model, concrete_args)
            trace = GraphModule(tracer.root, graph)
        else:
            raise Exception("model must be a torch.nn.Module or a torch.fx.GraphModule")

        return trace

    def print_tabular(self):
        node_specs = [[n.op, n.name, n.target, n.args, n.kwargs] for n in self.nodes]
        logger.debug(
            tabulate(
                node_specs,
                headers=["\nopcode", "\nname", "\ntarget", "\nargs", "\nkwargs"],
            )
        )

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

    def prepare(self):
        """
        Return:
            A GraphModule with observer (configured by qconfig_dict), ready for calibration
        """
        graph_module = self.get_graph_module(self.model, self.concrete_args)
        modules = dict(graph_module.named_modules())
        for node in list(graph_module.graph.nodes):
            if node.op == "placeholder":
                pass
            elif node.op == "call_module":
                module = modules[node.target]
                if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                    module.qconfig = get_qconfig(8)
                    module.qbit = 8
                    module.add_module(
                        "activation_pre_process", module.qconfig.activation()
                    )
                    module.register_forward_pre_hook(activation_pre_hook)
                    module.add_module(
                        "activation_post_process", module.qconfig.output()
                    )
                    module.register_forward_hook(activation_post_hook)
                elif isinstance(module, nn.ReLU):
                    module.qconfig = get_qconfig(8, output_dtype=torch.quint8)
                    module.qbit = 8
                    module.add_module(
                        "activation_pre_process", module.qconfig.activation()
                    )
                    module.register_forward_pre_hook(activation_pre_hook)
                    module.add_module(
                        "activation_post_process", module.qconfig.output()
                    )
                    module.register_forward_hook(activation_post_hook)
                elif isinstance(module, nn.Linear):
                    module.qconfig = get_qconfig(8)
                    module.qbit = 8
                    module.add_module(
                        "activation_pre_process", module.qconfig.activation()
                    )
                    module.register_forward_pre_hook(activation_pre_hook)
                    module.add_module(
                        "activation_post_process", module.qconfig.output()
                    )
                    module.register_forward_hook(activation_post_hook)
            elif node.op == "output":
                pass

        self.observed_model = torch.fx.GraphModule(graph_module, graph_module.graph)

    def calibrate(self, calibtraion_func):
        calibtraion_func(self.observed_model)

        logger.info("calibtraion finish")

    def convert(self):
        modules = dict(self.observed_model.named_modules())
        for node in list(self.observed_model.graph.nodes):
            if node.op == "placeholder":
                next_node = node.next
                module = modules[next_node.target]
                input = Input.from_float(module)
                with self.observed_model.graph.inserting_after(node):
                    self.observed_model.add_module("input", input)
                    new_node = self.observed_model.graph.call_module("input", (node,))

                next_node.replace_input_with(node, new_node)
            elif node.op == "call_module":
                module = modules[node.target]
                if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                    quantized = Conv2d.from_float(module)
                elif isinstance(module, nn.ReLU):
                    quantized = ReLU.from_float(module)
                elif isinstance(module, nn.MaxPool2d):
                    quantized = MaxPool2d.from_float(module)
                elif isinstance(module, nn.Linear):
                    quantized = Linear.from_float(module)

                with self.observed_model.graph.inserting_after(node):
                    self.observed_model.add_module(node.name, quantized)
                    new_node = self.observed_model.graph.call_module(
                        node.name, node.args, node.kwargs
                    )
                    node.replace_all_uses_with(new_node)
                    self.observed_model.graph.erase_node(node)

            elif node.op == "output":
                prev_node = node.args[0]
                module = dict(self.observed_model.named_modules())[prev_node.target]
                output = Output.from_float(module)
                with self.observed_model.graph.inserting_after(prev_node):
                    self.observed_model.add_module("output", output)
                    new_node = self.observed_model.graph.call_module(
                        "output", node.args, node.kwargs
                    )

                node.replace_input_with(prev_node, new_node)

        self.quanted_model = torch.fx.GraphModule(
            self.observed_model, self.observed_model.graph
        )

        logger.info("quantization finish")

    def evaluate(self, evaluate_func):
        evaluate_func(self.quanted_model)

    def profile(self, save_to_disk=False):
        dummy_input = self.gen_input_tensor(self.input_shape)

        logger.info("float profile")
        prof_float = FXProfiler(self.graph_module)
        for _ in range(10):
            prof_float.run(*dummy_input)

        prof_float.profiler.summary(save_to_disk)

        logger.info("quantized profile")
        prof_quant = FXProfiler(self.quanted_model)
        for _ in range(10):
            prof_quant.run(*dummy_input)

        prof_quant.profiler.summary(save_to_disk)

        import os

        def print_model_size(mdl):
            torch.save(mdl.state_dict(), "tmp.pt")
            print("%.2f MB" % (os.path.getsize("tmp.pt") / 1e6))
            os.remove("tmp.pt")

        print_model_size(self.graph_module)
        print_model_size(self.quanted_model)
