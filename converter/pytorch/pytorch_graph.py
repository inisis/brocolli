#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
import re
from converter.core.graph import GraphNode, Graph
import torch
import torch.jit
import torch.autograd
import torch.serialization
import contextlib
from torch.jit import _unique_state_dict

class PytorchGraphNode(GraphNode):

    def __init__(self, node, node_id, weights_name, output_shape):
        self._name = node.scopeName()
        self._kind = node.kind()
        self.id = node_id
        self.output_shape = output_shape
        super(PytorchGraphNode, self).__init__(node)

        self.attrs = {k : node[k] for k in node.attributeNames()}
        self.weights_name = weights_name

    @property
    def name(self):
        name = self._name + self.id
        name = name.replace('-','n').replace('\\','n').replace('/','n').replace('_','n').replace('[','n').replace(']','n')
        return name

    @property
    def type(self):
        return self._kind

    @property
    def output_ids(self):
        node_id = re.findall(r"%([\d]+) :", self.node.__str__())
        return node_id


class scope_name_workaround(object):
    def __init__(self):
        self.backup = None

    def __enter__(self):
        def _tracing_name(self_, tracing_state):
            if not tracing_state._traced_module_stack:
                return None
            module = tracing_state._traced_module_stack[-1]
            for name, child in module.named_modules():
                if child is self_:
                    return name
            return None

        def _slow_forward(self_, *input, **kwargs):
            tracing_state = torch._C._get_tracing_state()
            if not tracing_state or isinstance(self_.forward, torch._C.ScriptMethod):
                return self_.forward(*input, **kwargs)
            if not hasattr(tracing_state, '_traced_module_stack'):
                tracing_state._traced_module_stack = []
            name = _tracing_name(self_, tracing_state)
            if name:
                tracing_state.push_scope('%s[%s]' % (self_._get_name(), name))
            else:
                tracing_state.push_scope(self_._get_name())
            tracing_state._traced_module_stack.append(self_)
            try:
                result = self_.forward(*input, **kwargs)
            finally:
                tracing_state.pop_scope()
                tracing_state._traced_module_stack.pop()
            return result

        self.backup = torch.nn.Module._slow_forward
        setattr(torch.nn.Module, '_slow_forward', _slow_forward)

    def __exit__(self, type, value, tb):
        setattr(torch.nn.Module, '_slow_forward', self.backup)


class PytorchGraph(Graph):

    def __init__(self, model, opset_version):
        super(PytorchGraph, self).__init__(model, opset_version)
        self.model = model
        self.model.eval()        
        self.opset_version = opset_version
        self.state_dict = _unique_state_dict(self.model)
        self.shape_dict = dict()
        self.weights_names = list()
        self.ids = list()

    def get_node_id(self, node):
        node_id = re.search(r"[\d]+", node.__str__())
        return node_id.group(0)

    def get_input_id(self, node):
        node_id = re.findall(r"%([\d]+) :", node.__str__())
        return node_id

    def rename_nodes(self, node, node_id):
        node_scope = node.scopeName()
        node_name = node_scope + node_id
        node_name = node_name.replace('-','n').replace('\\','n').replace('/','n').replace('_','n').replace('[','n').replace(']','n')
        return node_name

    def extract(self, dummy_input, opset_version):
        with scope_name_workaround():
            torch.onnx.symbolic_helper._set_opset_version(opset_version)        
            trace_graph, torch_out, inputs_states = \
                torch.jit._get_trace_graph(self.model, (dummy_input, ),  strict=False, _force_outplace=False, _return_inputs_states=True)
            torch.onnx.utils.warn_on_static_input_change(inputs_states)

            nodes = list(trace_graph.nodes())
            scopename = []
            for node in nodes:
                scopename.append('.'.join(
                        re.findall(r'\[([\w\d.]+)\]', node.scopeName())
                    ))
            self.scope_name = list(dict.fromkeys(scopename))
            trace_graph = torch.onnx.utils._optimize_graph(trace_graph, torch.onnx.OperatorExportTypes.ONNX, params_dict={})

        nodes = list(trace_graph.nodes())
        for idx, node in enumerate(nodes):
            node_id = self.get_node_id(node)
            self.ids.append(node_id)
            node_name = 'node' + node_id
            if '.'.join(re.findall(r'\[([\w\d.]+)\]', node.scopeName())) == '':
                if (self.scope_name.index(self.weights_names[-1])) == len(self.scope_name) - 1:
                    self.weights_names.append(self.scope_name[self.scope_name.index(self.weights_names[-1])])
                else:
                    self.weights_names.append(self.scope_name[self.scope_name.index(self.weights_names[-1]) + 1])
            else:
                self.weights_names.append('.'.join(re.findall(r'\[([\w\d.]+)\]', node.scopeName())))

        return trace_graph, nodes

    def build(self, shape, opset_version):
        if isinstance(shape, tuple):
            dummy_input = []
            for each in shape:
                dummy = torch.ones(each)
                dummy_input.append(dummy)
            graph, nodes = self.extract(dummy_input, opset_version)
        else:
            dummy_input = torch.ones(shape)
            graph, nodes = self.extract(dummy_input, opset_version)

        for node, node_id, weight_name in zip(nodes, self.ids, self.weights_names):
            node_name = self.rename_nodes(node, node_id)
            output_str = node.__str__().split('=')[0]
            output_shape_str = re.findall(r'[^()!]+', output_str)
            if len(output_shape_str) > 1:
                output_shape = [int(x.replace('!', '')) if x.strip().isdigit() else None for x in output_shape_str[1].split(',')]
                output_shape = list(filter(None, output_shape))
            else:
                output_shape = None
            self.shape_dict[node_name] = output_shape
            self.layer_map[node_name] = PytorchGraphNode(node, node_id, weight_name, output_shape)
            self.layer_name_map[node_name] = node_name
            # make connection
            self.node_connection(graph, node, node_name)       

        super(PytorchGraph, self).build() 

    def node_connection(self, graph, node, node_name):
        for node_input in list(node.inputs()):
            node_input = node_input.node()
            node_id = self.get_node_id(node_input)
            if node_id:
                if node_input.scopeName() == '':
                    if node_id == '1':
                        continue
                    else:
                        input_ids = self.get_input_id(node_input)
                        if len(input_ids) == 1:
                            assert ("%" + input_ids[0]) in node_input.__str__()
                            node_input_name = self.rename_nodes(node_input, input_ids[0])
                            self._make_connection(node_input_name, node_name)
                        else:
                            for input_id in input_ids:
                                if ("%" + input_id) in node.__str__():
                                    input_id = input_ids[0] + ':' + input_id
                                    node_input_name = self.rename_nodes(node_input, input_id)
                                    self._make_connection(node_input_name, node_name)
                else:
                    input_ids = self.get_input_id(node_input)
                    if len(input_ids) == 1:
                        assert ("%" + input_ids[0]) in node_input.__str__()
                        node_input_name = self.rename_nodes(node_input, input_ids[0])
                        self._make_connection(node_input_name, node_name)
                    else:
                        for input_id in input_ids:
                            if ("%" + input_id) in node.__str__():
                                input_id = input_ids[0] + ':' + input_id
                                node_input_name = self.rename_nodes(node_input, input_id)
                                self._make_connection(node_input_name, node_name)
                    