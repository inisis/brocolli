#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from converter.core.graph import GraphNode, Graph
import torch
import torch.jit
import torch.autograd
import torch.serialization
import contextlib
from torch.jit import _unique_state_dict

class PytorchGraphNode(GraphNode):

    def __init__(self, layer):
        self._name = layer.scopeName()
        self._kind = layer.kind()
        import re
        node_id = re.search(r"[\d]+", layer.__str__())
        self.id = node_id.group(0)

        super(PytorchGraphNode, self).__init__(layer)
        if "L2Norm" not in self.name:
            self.attrs = {k : layer[k] for k in layer.attributeNames()}

        self.weights_name = '.'.join(
            re.findall(r'\[([\w\d.]+)\]', self._name)
        )

    @property
    def name(self):
        name = self._name + self.id
        # Scopes created in a nested scope may have initial characters
        # that are illegal as the initial character of an op name
        # (viz. '-', '\', '/', and '_').
        name = name.replace('-','n').replace('\\','n').replace('/','n').replace('_','n').replace('[','n').replace(']','n')
        return name

    @property
    def type(self):
        return self._kind

    @property
    def pytorch_layer(self):
        return self.layer


class PytorchGraph(Graph):

    def __init__(self, model):
        # sanity check.
        super(PytorchGraph, self).__init__(model)
        self.model = model
        self.state_dict = _unique_state_dict(self.model)
        self.shape_dict = dict()

    @staticmethod
    def _optimize_graph(graph, aten, export_raw_ir=False):
        # run dce first to eliminate dead parts of the graph that might have been
        # left behind by things like symbolic_override

        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)

        torch._C._jit_pass_peephole(graph)
        torch._C._jit_pass_lint(graph)
        if not export_raw_ir:
            graph = torch._C._jit_pass_onnx(graph, aten)
            torch._C._jit_pass_lint(graph)
            torch._C._jit_pass_onnx_peephole(graph)
            torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)
        graph = torch._C._jit_pass_canonicalize(graph)
        torch._C._jit_pass_lint(graph)
        return graph


    @staticmethod
    def get_node_id(node):
        import re
        node_id = re.search(r"[\d]+", node.__str__())
        return node_id.group(0)

    @contextlib.contextmanager
    def set_training(self, model, mode):
        r"""
        A context manager to temporarily set the training mode of 'model'
        to 'mode', resetting it when we exit the with-block.  A no-op if
        mode is None.
        """
        if mode is None:
            yield
            return
        old_mode = model.training
        if old_mode != mode:
            model.train(mode)
        try:
            yield
        finally:
            if old_mode != mode:
                model.train(old_mode)


    def build(self, shape):
        """
        build graph for pytorch 0.4.0
        """
        import re
        # construct graph
        dummy_input = torch.autograd.Variable(torch.randn(shape), requires_grad=False)

        with self.set_training(self.model, False):
            trace, output = torch.jit.get_trace_graph(self.model, (dummy_input, ))

        trace.set_graph(PytorchGraph._optimize_graph(trace.graph(), False))
        # nodes
        nodes = list(trace.graph().nodes())

        # input layer
        # TODO

        # build each layer
        flag_l2norm = False

        for node in nodes:

            if "L2Norm" in node.__str__():
                if 'SSDnL2NormnL2Normn96' in self.shape_dict.keys():  # exist
                    continue

                for k in self.shape_dict.keys():
                    node_id = PytorchGraph.get_node_id(node)
                    if str(int(node_id) - 1) in k:
                        output_shape = self.shape_dict[k]
                        node_input_name = k

                node_scope = node.scopeName()
                node_name = node_scope + node_id
                node_name = node_name.replace('-','n').replace('\\','n').replace('/','n').replace('_','n').replace('[','n').replace(']','n')

                self.shape_dict[node_name] = output_shape
                self.layer_map[node_name] = PytorchGraphNode(node)
                self.layer_name_map[node_name] = node_name

                self._make_connection(node_input_name, node_name)
                flag_l2norm = True
            else:
                node_id = PytorchGraph.get_node_id(node)
                node_scope = node.scopeName()
                node_name = node_scope + node_id
                node_name = node_name.replace('-','n').replace('\\','n').replace('/','n').replace('_','n').replace('[','n').replace(']','n')
                output_shape_str = re.findall(r'[^()!]+', node.__str__())[1]
                output_shape = [int(x.replace('!', '')) for x in output_shape_str.split(',')]

                self.shape_dict[node_name] = output_shape
                self.layer_map[node_name] = PytorchGraphNode(node)
                self.layer_name_map[node_name] = node_name

                # input
                if flag_l2norm:
                    self._make_connection('SSDnL2NormnL2Normn96', node_name)
                    flag_l2norm = False
                    continue

                for node_input in list(node.inputs()):
                    if PytorchGraph.get_node_id(node_input.node()) == '107':
                        self._make_connection('SSDnL2NormnL2Normn96', node_name)
                        continue

                for node_input in list(node.inputs()):

                    if PytorchGraph.get_node_id(node_input.node()) and node_input.node().scopeName():
                        node_input_name = node_input.node().scopeName() + PytorchGraph.get_node_id(node_input.node())
                        node_input_name = node_input_name.replace('-','n').replace('\\','n').replace('/','n').replace('_','n').replace('[','n').replace(']','n')
                        self._make_connection(node_input_name, node_name)
                        # print(node_input_name ,'->', node_name)


        super(PytorchGraph, self).build()
