#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
import logging
import numpy as np
from converter.core.parser import Parser
from converter.pytorch.pytorch_graph import PytorchGraph
import caffe.proto.caffe_pb2 as pb2

import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

import collections
import tensorrt as trt


layer_map = {
    'Data': 'Data',
    'onnx::Conv': 'Conv',
    'onnx::Sigmoid': 'Sigmoid',
    'onnx::PRelu': 'PRelu',
    'onnx::BatchNormalization': 'BatchNormalization',
    'onnx::Relu': 'Relu',
    'onnx::MaxPool': 'MaxPool',
    'onnx::Add': 'Add',
    'onnx::AveragePool': 'AveragePool',
    'onnx::GlobalAveragePool': 'GlobalAveragePool',
    'onnx::Flatten': 'Flatten',
    'onnx::Gemm': 'FullyConnected',
    'onnx::Dropout': 'Dropout',
    'onnx::LogSoftmax': 'Softmax',
    'onnx::Transpose': 'Permute',
    'onnx::Upsample': 'Upsample',
    'onnx::Concat': 'Concat',
    'onnx::Unsqueeze': "Unsqueeze",
    'onnx::Clip': "Relu6",
    'onnx::Pad': "Pad",
    'onnx::HardSwish': "HardSwish",
    'onnx::HardSigmoid': "HardSigmoid",
    'onnx::Mul': 'Mul',    
    'onnx::Slice': 'Slice', 
    'onnx::Softmax': 'Softmax',
    'onnx::Constant': 'Common',
    'onnx::Reshape': 'Reshape',
    'onnx::Split': 'Split',
    'onnx::LpNormalization': 'LpNormalization',
    'prim::Constant': 'Constant',
    'onnx::LeakyRelu': 'LeakyRelu',
    'onnx::Resize': 'Resize',
    'onnx::ReduceMean': 'ReduceMean',
    'onnx::BilinearInterpolate': 'BilinearInterpolate',
    'onnx::Shape': 'Common',
    'onnx::Gather': 'Common',
    'onnx::Sub': 'Common',
    'onnx::MaxUnpool': 'MaxUnPool',
    'onnx::ConvTranspose': 'ConvTranspose',
    'onnx::Cast': 'Common',
    'onnx::ConstantOfShape': 'Common',
    'onnx::Div': 'Common'
}

class PytorchTensorRTParser(Parser):
    def __init__(self, model, input_shape, opset_version, fuse=False):
        super(PytorchTensorRTParser, self).__init__()
        self.fuse = fuse
        self.model = model
        if self.fuse:
            self.fuse_all_conv_bn(self.model)
        self.pytorch_graph = PytorchGraph(self.model, opset_version)
        self.input_shape = input_shape
        self.opset_version = opset_version
        self.pytorch_graph.build(self.input_shape, self.opset_version)
        self.state_dict = self.pytorch_graph.state_dict
        self.shape_dict = self.pytorch_graph.shape_dict
        self.named_layer = dict()
        self.named_node = dict()
        self.tensorrt_net = collections.OrderedDict()
        self.main_layers = []

    def run(self, dest_path):
        engine = self.gen_IR()
        self.save_to_file(engine, dest_path + ".trt")

    def save_to_file(self, engine, filename):
        with open(filename, 'wb') as f:
            f.write(engine.serialize())

    @property
    def src_graph(self):
        return self.pytorch_graph

    def fuse_all_conv_bn(self, model):
        stack = []
        for name, module in model.named_children():
            if list(module.named_children()):
                self.fuse_all_conv_bn(module)
                
            if isinstance(module, nn.BatchNorm2d):
                if not stack:
                    continue
                if isinstance(stack[-1][1], nn.Conv2d):
                    setattr(model, stack[-1][0], fuse_conv_bn_eval(stack[-1][1], module))
                    setattr(model, name, nn.Identity())
            else:
                stack.append((name, module))

    def is_main(self, inputs):
        for input in inputs:
            find = False
            for layer in self.main_layers:
                if input in layer.top:
                    find = True
                    break

            if find == False:
                return False

        return True

    def gen_IR(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.builder = trt.Builder(TRT_LOGGER)
        self.network = self.builder.create_network()
        self.config = self.builder.create_builder_config()
        self.runtime = trt.Runtime(TRT_LOGGER)        
        self.config.max_workspace_size = (1 << 30)
        self.builder.max_batch_size = 1

        for node in self.src_graph.topological_sort:
            current_node = self.src_graph.get_node(node)
            self.named_node[current_node.real_name] = current_node
            onnx_node_type = current_node.type
            node_type = layer_map[onnx_node_type]

            if hasattr(self, "rename_" + node_type):
                func = getattr(self, "rename_" + node_type)
                layer_data = func(current_node)
                if layer_data == None:
                    continue
                elif(isinstance(layer_data, tuple)):
                    self.named_layer[layer_data[0].name] = layer_data[0]
                    self.named_layer[layer_data[1].name] = layer_data[1] # some batchnorm will not be eliminated
                else:                  
                    self.named_layer[layer_data.name] = layer_data
            else:
                self.rename_Common(current_node)

        for layer_name in self.pytorch_graph.output_layers:
            self.network.mark_output(tensor=self.tensorrt_net[layer_name].get_output(0))              

        self.plan = self.builder.build_serialized_network(self.network, self.config)
        engine = self.runtime.deserialize_cuda_engine(self.plan)
        return engine

    ##########
    # Layers #
    ##########
    def rename_Common(self, source_node):
        logging.warning("PyTorch parser will skip operator [%s] with name [%s]."
              % (source_node.type, source_node.name)) 

        return None

    def rename_Data(self, source_node):
        layer = self.network.add_input(source_node.name, dtype=trt.float32, shape=source_node.output_shape)
        assert(layer)
        self.tensorrt_net["data"] = layer        
       
        return layer

    def rename_Conv(self, source_node):
        attr = source_node.attrs
        if len(attr['pads']) == 4:
            pads = (attr['pads'][0], attr['pads'][1])
        elif len(attr['pads']) == 2:
            pads = (attr['pads'][0], attr['pads'][1])

        if 'strides' not in attr:
            strides = None
        else:
            strides = (attr['strides'][0], attr['strides'][1])

        if 'kernel_shape' not in attr:
            kernel_shape = None 
        else:
            kernel_shape = (attr['kernel_shape'][0], attr['kernel_shape'][1])

        bias_name = '{0}.bias'.format(source_node.weights_name)
        weights_name = '{0}.weight'.format(source_node.weights_name)
        weight = self.state_dict[weights_name]

        weight = weight.numpy()

        self.set_weight(source_node.name, 'weights', weight)
        num_filter = list(weight.shape)[0]

        # handle bias
        if bias_name in self.state_dict:
            bias = self.state_dict[bias_name].numpy()
        else:
            trt.Weights()

        if isinstance(self.tensorrt_net[source_node.in_edges[0]], trt.tensorrt.ITensor):
            input_ = self.tensorrt_net[source_node.in_edges[0]]
        else:
            input_ = self.tensorrt_net[source_node.in_edges[0]].get_output(0)

        layer = self.network.add_convolution(input=input_, num_output_maps=num_filter, kernel_shape=kernel_shape, kernel=weight, bias=bias)    
        layer.stride = strides
        layer.padding = pads

        self.tensorrt_net[source_node.name] = layer

        return layer

    def rename_Relu(self, source_node):
        layer = self.network.add_activation(self.tensorrt_net[source_node.in_edges[0]].get_output(0), type=trt.ActivationType.RELU)
        self.tensorrt_net[source_node.name] = layer

        return layer

    def rename_MaxPool(self, source_node):
        attr = source_node.attrs
        kwargs = dict()

        if len(attr['pads']) == 4:
            pads = (attr['pads'][0], attr['pads'][1])
        elif len(attr['pads']) == 2:
            pads = (attr['pads'][0], attr['pads'][1])

        if 'strides' not in attr:
            kwargs['strides'] = [1] + [1, 1] + [1]
            layer.pooling_param.stride = 1
        else:
            strides = (attr['strides'][0], attr['strides'][1]) 

        if 'kernel_shape' not in attr:
            kwargs['kernel_shape'] = [1] + [1, 1] + [1]
            layer.pooling_param.kernel_size.extend(1)
        else:
            kwargs['kernel_shape'] = [1] + attr['kernel_shape'] + [1]
            kernel_shape = (attr['kernel_shape'][0], attr['kernel_shape'][1])            

        layer = self.network.add_pooling(self.tensorrt_net[source_node.in_edges[0]].get_output(0), trt.PoolingType.MAX, window_size=kernel_shape)
        layer.stride = strides
        layer.padding = pads

        self.tensorrt_net[source_node.name] = layer

        return layer

    def rename_Add(self, source_node):
        layer = self.network.add_elementwise(self.tensorrt_net[source_node.in_edges[0]].get_output(0), self.tensorrt_net[source_node.in_edges[1]].get_output(0), trt.ElementWiseOperation.SUM)
        self.tensorrt_net[source_node.name] = layer

        return layer

    def rename_GlobalAveragePool(self, source_node):
        shape  = self.tensorrt_net[source_node.in_edges[0]].get_output(0).shape
        layer = self.network.add_pooling(self.tensorrt_net[source_node.in_edges[0]].get_output(0), trt.PoolingType.AVERAGE, window_size=shape[2:])
        self.tensorrt_net[source_node.name] = layer

        return layer

    def rename_Flatten(self, source_node):
        self.tensorrt_net[source_node.name] = self.tensorrt_net[source_node.in_edges[0]]
        return

    def rename_FullyConnected(self, source_node):
        bias_name = '{0}.bias'.format(source_node.weights_name)
        weights_name = '{0}.weight'.format(source_node.weights_name)

        W = self.state_dict[weights_name].numpy().transpose()

        _, output_channels = W.shape

        weight = self.state_dict[weights_name].numpy()

        self.set_weight(source_node.name, 'weights', W )

        # use_bias
        if bias_name in self.state_dict:
            bias = self.state_dict[bias_name].numpy()
        else:
            bias = trt.Weights()

        layer = self.network.add_fully_connected(input=self.tensorrt_net[source_node.in_edges[0]].get_output(0), num_outputs=output_channels, kernel=weight, bias=bias)
        
        self.tensorrt_net[source_node.name] = layer

        return layer

    def rename_Sigmoid(self, source_node):
        layer = self.network.add_activation(self.tensorrt_net[source_node.in_edges[0]].get_output(0), type=trt.ActivationType.SIGMOID)
        layer.name = source_node.real_name

        self.tensorrt_net[source_node.name] = layer

        return layer

    def rename_ReduceSum(self, source_node):
        attr = source_node.attrs
        axes = np.power(2, attr['axes'][0]) # bit wise 
        layer = self.network.add_reduce(self.tensorrt_net[source_node.in_edges[0]].get_output(0), trt.ReduceOperation.SUM, axes=axes, keep_dims=True)
        self.tensorrt_net[source_node.name] = layer

        return layer

    def rename_Div(self, source_node):
        layer = self.network.add_elementwise(self.tensorrt_net[source_node.in_edges[0]].get_output(0), self.tensorrt_net[source_node.in_edges[1]].get_output(0), trt.ElementWiseOperation.DIV)
        self.tensorrt_net[source_node.name] = layer
        print("div : ", layer.get_output(0).shape)
        print(source_node.name)
        
        return layer

    def rename_Mul(self, source_node):
        layer = self.network.add_elementwise(self.tensorrt_net[source_node.in_edges[0]].get_output(0), self.tensorrt_net[source_node.in_edges[1]].get_output(0), trt.ElementWiseOperation.PROD)
        self.tensorrt_net[source_node.name] = layer

        return layer

    def rename_ConvTranspose(self, source_node):
        attr = source_node.attrs
        if len(attr['pads']) == 4:
            pads = (attr['pads'][0], attr['pads'][1])
        elif len(attr['pads']) == 2:
            pads = (attr['pads'][0], attr['pads'][1])

        if 'strides' not in attr:
            strides = None
        else:
            strides = (attr['strides'][0], attr['strides'][1])

        if 'kernel_shape' not in attr:
            kernel_shape = None
        else:
            kernel_shape = (attr['kernel_shape'][0], attr['kernel_shape'][1])

        bias_name = '{0}.bias'.format(source_node.weights_name)
        weights_name = '{0}.weight'.format(source_node.weights_name)

        weight = self.state_dict[weights_name]
        weight = weight.numpy()

        num_groups = list(weight.shape)[0]

        # handle bias
        if bias_name in self.state_dict:
            bias = self.state_dict[bias_name].numpy()
        else:
            bias = trt.Weights()

        layer = self.network.add_deconvolution(input=self.tensorrt_net[source_node.in_edges[0]].get_output(0), num_output_maps=num_groups,
                                        kernel_shape=kernel_shape, kernel=weight, bias=bias)
        layer.stride = strides
        layer.padding = pads

        self.tensorrt_net[source_node.name] = layer

        return layer
