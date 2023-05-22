import re
import sys
import copy
from loguru import logger
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.fx
from torch.fx import Tracer, Graph, Node
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.graph_module import GraphModule

from .fuser import is_match
from .profiler import FXProfiler
from .comparator import FXComparator
from .qconfig import get_qconfig
from .pattern import get_default_fusion_patterns
from .utils import (
    replace_node_module,
    check_result,
    _node_dict,
    create_target,
    get_function_name,
)
from .graph_modules import BrocolliGraphModule
from .observer import get_available_observers

from .quantization_layers import *
from .quantization_layers.registry import get_default_quant_ops


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
    def __init__(self, model, input_shape, concrete_args=None, log_level=1):
        super(PytorchQuantizer, self).__init__()
        self.model = model.eval()
        self.device = next(self.model.parameters()).device
        self.input_shape = input_shape
        if isinstance(input_shape, (tuple, list)) and all(
            isinstance(element, int) for element in input_shape
        ):
            self.input_shape = [input_shape]
        self.concrete_args = concrete_args
        self.qconfig = None
        self.qconfig_dict = {"": self.qconfig}
        self.init_logging(log_level)
        self.graph_module = self.get_graph_module(self.model, self.concrete_args, False)
        self.modules = dict(self.graph_module.named_modules())
        self.print_tabular(self.graph_module)
        self.quant_ops = get_default_quant_ops()

    def init_logging(self, log_level):
        logger.remove()
        if log_level == 0:
            logger.add(sys.stderr, level="DEBUG")
        elif log_level == 1:
            logger.add(sys.stderr, level="INFO")
        elif log_level == 2:
            logger.add(sys.stderr, level="WARNING")
        elif log_level == 3:
            logger.add(sys.stderr, level="ERROR")
        else:
            raise Exception("level must be 0, 1, 2 or 3")

    def get_graph_module(self, model, concrete_args, inplace=True):
        if not inplace:
            model = copy.deepcopy(model)

        if isinstance(model, GraphModule):
            graph_module = BrocolliGraphModule(model.root, model.graph)
        elif isinstance(model, nn.Module):
            tracer = BrocolliTracer()
            graph = tracer.trace(model, concrete_args)
            graph_module = BrocolliGraphModule(tracer.root, graph)
        else:
            raise Exception("model must be a torch.nn.Module or a torch.fx.GraphModule")

        return graph_module

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
                    input_tensor.append(
                        torch.rand(shape).to(torch.float32).to(self.device)
                    )
                else:
                    input_tensor.append(self.gen_input_tensor(shape))
            else:
                input_tensor.append(torch.rand(shape).to(torch.float32).to(self.device))

        return input_tensor

    def shape_inference(self):
        dummy_input = self.gen_input_tensor(self.input_shape)
        ShapeProp(self.graph_module).propagate(*dummy_input)

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

        fused_model = BrocolliGraphModule(graph_module, fused_graph)
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

    def _is_quant_op(self, node, graph_module, quant_ops):
        modules = dict(graph_module.named_modules())
        if node.op == "placeholder":
            return True
        elif node.op == "call_module":
            module = modules[node.target]
            for op, _ in quant_ops.items():
                if not isinstance(op, str) and isinstance(module, op):
                    return True
        elif node.op == "call_function":
            function_name = get_function_name(node.target)
            if function_name == "add":
                return True
            else:
                return False
        elif node.op == "output":
            return False
        else:
            return False

    def _is_observer_needed(self, node, graph_module, quant_ops, lsq):
        modules = dict(graph_module.named_modules())
        if node.op == "placeholder":
            if lsq:
                return False
            else:
                return True
        elif node.op == "call_module":
            user = list(node.users)[0]
            if user.op == "call_module":
                user_module = modules[user.target]
                if isinstance(user_module, nn.ReLU):
                    return False

            module = modules[node.target]
            if isinstance(module, nn.MaxPool2d):
                return False
            else:
                return True

        elif node.op == "call_function":
            user = list(node.users)[0]
            if user.op == "call_module":
                user_module = modules[user.target]
                if isinstance(user_module, nn.ReLU):
                    return False

            function_name = get_function_name(node.target)
            if function_name == "add":
                return False
            else:
                return False
        elif node.op == "output":
            return False
        else:
            return False

    def prepare(self, lsq=False):
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
            if node.op == "call_module":
                module = modules[node.target]
                target_atoms = node.target.split(".")
                target = "_".join(target_atoms)
                match = re.findall(r"{}(_+\d)".format(str(target)), node.name)
                if match:
                    module = copy.deepcopy(module)
                    with graph_module.graph.inserting_after(node):
                        node_target = str(node.target) + match[0]
                        graph_module.add_submodule(node_target, module)
                        node.target = node_target

            if self._is_observer_needed(node, graph_module, self.quant_ops, lsq):
                users = list(node.users)
                qconfig = get_qconfig(8, lsq=lsq)
                observer_module = qconfig.activation()
                observer_module.qconfig = qconfig
                observer_module.qbit = 8
                with graph_module.graph.inserting_after(node):
                    graph_module.add_module(node.name + "_observer", observer_module)
                    new_node = graph_module.graph.call_module(
                        node.name + "_observer", (node,), type_expr="observer"
                    )
                for user in users:
                    user.replace_input_with(node, new_node)

        self.observed_model = torch.fx.GraphModule(graph_module, graph_module.graph)

    def finetune(self, train_func):
        for name, param in self.observed_model.named_parameters():
            if not "observer" in name:
                param.requires_grad_(False)

        train_func(self.observed_model)
        logger.info("calibtraion finish")

    def calibrate(self, calibtraion_func):
        self.calibrate_func = calibtraion_func
        logger.info("calibtraion start")
        calibtraion_func(self.observed_model)
        logger.info("calibtraion finish")

    def find_input_observer_node(self, node):
        if node.type == "observer":
            return node
        else:
            return self.find_input_observer_node(node.all_input_nodes[0])

    def find_output_observer_node(self, node):
        if node.type == "observer":
            return node
        else:
            return self.find_output_observer_node(list(node.users)[0])

    def convert(self):
        graph_module = copy.deepcopy(self.observed_model)
        modules = dict(graph_module.named_modules())
        self.op_maps = {}
        for node in list(graph_module.graph.nodes):
            if node.op == "placeholder":
                next_node = node.next
                module = modules[next_node.target]
                input_module = Input.from_float(module)
                with graph_module.graph.inserting_after(node):
                    graph_module.add_module("input", input_module)
                    new_node = graph_module.graph.call_module("input", (node,))

                next_node.replace_input_with(node, new_node)
                self.op_maps[new_node.name] = node.name
            elif (
                node.op == "call_module"
                and node.type != "observer"
                and self._is_quant_op(node, graph_module, self.quant_ops)
            ):
                module = modules[node.target]
                assert len(node.all_input_nodes) == 1
                input_observer_node = self.find_input_observer_node(node)
                module.activation_pre_process = modules[input_observer_node.target]
                output_observer_node = self.find_output_observer_node(node)
                module.activation_post_process = modules[output_observer_node.target]
                self.op_maps[node.name] = output_observer_node.name[:-9]
                if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                    quantized = Conv2d.from_float(module)
                elif isinstance(module, nn.ReLU):
                    quantized = ReLU.from_float(module)
                elif isinstance(module, nn.MaxPool2d):
                    quantized = MaxPool.from_float(module)
                elif isinstance(module, nn.Linear):
                    quantized = Linear.from_float(module)
                elif isinstance(module, nn.AdaptiveMaxPool2d):
                    quantized = MaxPool.from_float(module)
                elif isinstance(module, nn.AdaptiveAvgPool2d):
                    quantized = AdaptiveAvgPool.from_float(module)
                elif isinstance(module, tuple(get_available_observers())):
                    logger.info("skip observer")
                else:
                    raise Exception(f"not supported module {module.__class__.__name__}")

                with graph_module.graph.inserting_after(node):
                    replace_node_module(node, modules, quantized)
            elif node.op == "call_function" and self._is_quant_op(
                node, graph_module, self.quant_ops
            ):
                function_name = get_function_name(node.target)
                if function_name == "add":
                    assert len(node.all_input_nodes) == 2
                    observer_node = self.find_input_observer_node(node.args[0])
                    node.activation_pre_process1 = modules[observer_node.target]
                    observer_node = self.find_input_observer_node(node.args[1])
                    node.activation_pre_process2 = modules[observer_node.target]
                    output_observer_node = self.find_output_observer_node(node)
                    node.activation_post_process = modules[output_observer_node.target]
                    quantized = Add.from_float(node)
                    with graph_module.graph.inserting_after(node):
                        graph_module.add_submodule(node.name + "_quanted", quantized)
                        quanted_node = graph_module.graph.call_module(
                            node.name + "_quanted", node.args
                        )
                    node.replace_all_uses_with(quanted_node)
                    graph_module.graph.erase_node(node)
                    quanted_node.name = node.name
                    self.op_maps[node.name] = output_observer_node.name[:-9]

            elif node.op == "output":
                if isinstance(node.args[0], Node):
                    prev_node = node.args[
                        0
                    ]  # previous node may not be observer node when not quanted
                    observer_node = self.find_input_observer_node(node.args[0])
                    module = dict(graph_module.named_modules())[observer_node.target]

                    output = Output.from_float(module)
                    with graph_module.graph.inserting_after(prev_node):
                        graph_module.add_module("output", output)
                        new_node = graph_module.graph.call_module(
                            "output", node.args, node.kwargs
                        )

                    node.replace_input_with(prev_node, new_node)
                else:
                    for idx in range(len(node.args[0])):
                        prev_node = node.args[0][idx]
                        observer_node = self.find_observer_node(node.args[0][idx])
                        module = dict(graph_module.named_modules())[
                            observer_node.target
                        ]

                        output = Output.from_float(module)
                        with graph_module.graph.inserting_after(prev_node):
                            graph_module.add_module("output_" + str(idx), output)
                            new_node = graph_module.graph.call_module(
                                "output_" + str(idx), (prev_node,)
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
        logger.info("evaluation start")
        evaluate_func(self.quanted_model)
        logger.info("evaluation finish")

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

    def compare(self, interrested_node=None):
        logger.info("float runs")
        if hasattr(self, "fused_model"):
            float_model = self.fused_model
        else:
            float_model = self.graph_module

        prof_float = FXComparator(float_model)
        self.calibrate_func(prof_float)

        logger.info("quantized runs")
        quanted_model = self.quanted_model
        prof_quant = FXComparator(quanted_model)
        self.calibrate_func(prof_quant)

        float_node_dict = _node_dict(float_model)
        quant_node_dict = _node_dict(quanted_model)

        def compare(float_node_dict, quant_node_dict, float_model, quanted_model):
            quanted_modules = dict(quanted_model.named_modules())
            node_compare_specs = []
            for quanted_name in self.op_maps.keys():
                float_name = self.op_maps[quanted_name]
                logger.debug(
                    f"compare float op : {float_name} and quanted op: {quanted_name}"
                )
                float_node = float_node_dict[float_name]
                quant_node = quant_node_dict[quanted_name]

                with torch.no_grad():
                    if quant_node.op == "call_module":
                        module = quanted_modules[quant_node.target]
                        float_data = float_node.meta["tensor_meta"]["tensor"].flatten()
                        quant_data = (
                            quant_node.meta["tensor_meta"]["tensor"].flatten()
                            * module.output_scale
                        )
                        cos_sim = F.cosine_similarity(float_data, quant_data, dim=0)
                        mre = (
                            torch.abs(quant_data - float_data).sum()
                            * 100.0
                            / torch.abs(float_data).sum()
                        )
                        if float_node.op == "placeholder":
                            quanted_name = float_node.name
                        node_compare_specs.append([quanted_name, cos_sim, mre])
                        if interrested_node is not None and quanted_name in interrested_node:
                            import plotext as plt
                            plt.theme('matrix')
                            plt.subplots(1, 2)
                            plt.subplot(1, 1)                     
                            plt.hist(float_data.numpy(), bins=256)
                            plt.subplot(1, 2)
                            plt.hist(quant_data.numpy(), bins=256)
                            plt.show()

            logger.info(
                tabulate(
                    node_compare_specs,
                    headers=["\nname", "\ncos_sim", "\nmre"],
                    floatfmt=".4f",
                )
            )

        compare(float_node_dict, quant_node_dict, float_model, quanted_model)
