import copy
from loguru import logger
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.fx
from torch.fx import Tracer, Graph
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.graph_module import GraphModule

from .fuser import is_match
from .profiler import FXProfiler
from .qconfig import get_qconfig
from .pattern import get_default_fusion_patterns
from .utils import (
    replace_node_module,
    check_result,
)
from .graph_modules import BrocolliGraphModule
from .observer import get_available_observers

from brocolli.quantization.quantization_layers import *


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

        self.graph_module = self.get_graph_module(self.model, self.concrete_args, False)
        self.modules = dict(self.graph_module.named_modules())
        self.print_tabular(self.graph_module)

    def get_graph_module(self, model, concrete_args, inplace=True):
        if not inplace:
            model = copy.deepcopy(model)

        if isinstance(model, GraphModule):
            trace = BrocolliGraphModule(model.root, model.graph)
        elif isinstance(model, nn.Module):
            tracer = BrocolliTracer()
            graph = tracer.trace(model, concrete_args)
            trace = BrocolliGraphModule(tracer.root, graph)
        else:
            raise Exception("model must be a torch.nn.Module or a torch.fx.GraphModule")

        return trace

    def print_tabular(self, graph_module):
        nodes = list(graph_module.graph.nodes)
        node_specs = [[n.op, n.name, n.target, n.args, n.kwargs] for n in nodes]
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

    def fuse(self):
        graph_module = self.get_graph_module(self.model, self.concrete_args, False)
        modules = dict(graph_module.named_modules())
        fusion_patterns = get_default_fusion_patterns()
        fusion_pairs = self._find_matches(
            graph_module, graph_module.graph, fusion_patterns
        )
        logger.debug("fusion_pairs: {}", fusion_pairs)
        fused_graph = graph_module.graph
        env = {}

        for node in graph_module.graph.nodes:
            root_node, obj = fusion_pairs.get(node.name, (None, None))
            if root_node is node:
                assert obj is not None
                env[node.name] = obj.fuse(fused_graph, modules)

        fused_model = BrocolliGraphModule(self.model, fused_graph)

        dummy_input = self.gen_input_tensor(self.input_shape)
        float_output = self.forward(fused_model, dummy_input)
        fused_output = self.forward(self.model, dummy_input)
        check_result(float_output, fused_output)
        self.fused_model = fused_model

    def forward(self, model, input):
        output = model(*input)

        if isinstance(output, torch.Tensor):
            output = [output]

        return output

    def _find_matches(self, root, graph, patterns):
        modules = dict(root.named_modules())
        match_map = {}

        def apply_match(pattern, node, match):
            if isinstance(pattern, tuple):
                s, *args = pattern
                apply_match(s, node, match)
                for subpattern, arg in zip(args, node.args):
                    apply_match(subpattern, arg, match)
            else:
                # the first pattern matches will take precedence
                if node.name not in match_map:
                    match_map[node.name] = match

        for node in reversed(graph.nodes):
            if node.name not in match_map:
                for pattern, value in patterns.items():
                    if is_match(modules, node, pattern):
                        logger.debug("matched patter {}", pattern)
                        apply_match(pattern, node, (node, value(self, node)))

        return match_map

    def prepare(self):
        """
        Return:
            A GraphModule with observer (configured by qconfig_dict), ready for calibration
        """
        if hasattr(self, "fused_model"):
            graph_module = copy.deepcopy(self.fused_model)
        else:
            graph_module = copy.deepcopy(self.graph_module)
        modules = dict(graph_module.named_modules())
        for node in list(graph_module.graph.nodes):
            if node.op == "placeholder":
                next_node = node.next
                qconfig = get_qconfig(8)
                observer_module = qconfig.activation()
                with graph_module.graph.inserting_after(node):
                    graph_module.add_module(node.name + "_observer", observer_module)
                    new_node = graph_module.graph.call_module(
                        node.name + "_observer", (node,), type_expr="observer"
                    )

                next_node.replace_input_with(node, new_node)
            elif node.op == "call_module":
                module = modules[node.target]
                next_node = node.next
                qconfig = get_qconfig(8)
                observer_module = qconfig.activation()
                module.qconfig = qconfig
                module.qbit = 8
                with graph_module.graph.inserting_after(node):
                    graph_module.add_module(node.name + "_observer", observer_module)
                    new_node = graph_module.graph.call_module(
                        node.name + "_observer", (node,), type_expr="observer"
                    )

                next_node.replace_input_with(node, new_node)
            elif node.op == "call_function":
                next_node = node.next
                qconfig = get_qconfig(8)
                observer_module = qconfig.activation()
                with graph_module.graph.inserting_after(node):
                    graph_module.add_module(node.name + "_observer", observer_module)
                    new_node = graph_module.graph.call_module(
                        node.name + "_observer", (node,), type_expr="observer"
                    )

                next_node.replace_input_with(node, new_node)
            elif node.op == "output":
                pass

        self.observed_model = torch.fx.GraphModule(graph_module, graph_module.graph)

    def calibrate(self, calibtraion_func):
        calibtraion_func(self.observed_model)

        logger.info("calibtraion finish")

    def convert(self):
        graph_module = copy.deepcopy(self.observed_model)
        modules = dict(graph_module.named_modules())
        for node in list(graph_module.graph.nodes):
            if node.op == "placeholder":
                next_node = node.next
                module = modules[next_node.target]
                input = Input.from_float(module)
                with graph_module.graph.inserting_after(node):
                    graph_module.add_module("input", input)
                    new_node = graph_module.graph.call_module("input", (node,))

                next_node.replace_input_with(node, new_node)
            elif node.op == "call_module" and node.type != "observer":
                module = modules[node.target]

                assert len(node.all_input_nodes) == 1
                for input_node in node.all_input_nodes:
                    if input_node.type == "observer":
                        module.activation_pre_process = modules[input_node.target]

                module.activation_post_process = modules[node.name + "_observer"]

                if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                    quantized = Conv2d.from_float(module)
                elif isinstance(module, nn.ReLU):
                    quantized = ReLU.from_float(module)
                elif isinstance(module, nn.MaxPool2d):
                    quantized = MaxPool2d.from_float(module)
                elif isinstance(module, nn.Linear):
                    quantized = Linear.from_float(module)
                elif isinstance(module, tuple(get_available_observers())):
                    continue

                with graph_module.graph.inserting_after(node):
                    replace_node_module(node, modules, quantized)

            elif node.op == "output":
                prev_node = node.args[0]
                module = dict(graph_module.named_modules())[prev_node.target]

                output = Output.from_float(module)
                with graph_module.graph.inserting_after(prev_node):
                    graph_module.add_module("output", output)
                    new_node = graph_module.graph.call_module(
                        "output", node.args, node.kwargs
                    )

                node.replace_input_with(prev_node, new_node)

        for node in list(graph_module.graph.nodes):  # remove observer
            if node.op == "call_module" and node.type == "observer":
                assert len(node.all_input_nodes) == 1
                input_node = node.all_input_nodes[0]
                node.replace_all_uses_with(input_node)
                graph_module.graph.erase_node(node)

        self.quanted_model = torch.fx.GraphModule(graph_module, graph_module.graph)
        self.print_tabular(self.quanted_model)
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
