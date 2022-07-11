#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
import numpy as np
from loguru import logger
from converter.core.parser import Parser
from converter.pytorch.jit.pytorch_graph import PytorchGraph
import caffe.proto.caffe_pb2 as pb2

import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval

import google.protobuf.text_format
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
    'onnx::Clip': "Clip",
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
    'onnx::Gather': 'Gather',
    'onnx::Sub': 'Common',
    'onnx::MaxUnpool': 'MaxUnPool',
    'onnx::ConvTranspose': 'ConvTranspose',
    'onnx::Cast': 'Common',
    'onnx::ConstantOfShape': 'Common',
    'onnx::Div': 'Common',
    'onnx::ArgMax': 'ArgMax'
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
        self.main_layers = []

    def run(self, dest_path):
        engine = self.gen_IR(dest_path)
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

            elif isinstance(module, nn.BatchNorm1d):
                if not stack:
                    continue                
                if isinstance(stack[-1][1], nn.Linear):
                    setattr(model, stack[-1][0], fuse_linear_bn_eval(stack[-1][1], module))
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

    def save_to_proto(self, net, filename):
        with open(filename, 'wb') as f:
            f.write(google.protobuf.text_format.MessageToString(net).encode())

    def gen_IR(self, dest_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.builder = trt.Builder(TRT_LOGGER)
        self.network = self.builder.create_network()
        self.config = self.builder.create_builder_config()
        self.runtime = trt.Runtime(TRT_LOGGER)        
        self.config.max_workspace_size = (1 << 30)
        self.builder.max_batch_size = 1

        try:
            for node in self.src_graph.topological_sort:
                current_node = self.src_graph.get_node(node)
                self.named_node[current_node.real_name] = current_node
                onnx_node_type = current_node.type
                node_type = layer_map[onnx_node_type]

                if hasattr(self, "rename_" + node_type):
                    func = getattr(self, "rename_" + node_type)
                    func(current_node)
                else:
                    self.rename_Common(current_node)
        except Exception as e:
            logger.info(e)
        finally:
            text_net = pb2.NetParameter()

            for layer in self.main_layers:
                layer.name = layer.name.replace(".", "")
                layer_proto = pb2.LayerParameter()
                layer_proto.CopyFrom(layer)
                del layer_proto.blobs[:]
                text_net.layer.extend([layer_proto])
            self.save_to_proto(text_net, dest_path + "_debug.prototxt")

        for layer_name in self.pytorch_graph.output_layers:
            node = self.named_node[layer_name]

            if node.type in ['onnx::Split']:
                for idx, output_id in enumerate(node.output_ids):     
                    output_name = node.name + ':' + output_id 
                    self.network.mark_output(tensor=self.named_layer[output_name])
            else:
                if layer_name not in self.named_layer.keys():
                    continue
                self.network.mark_output(tensor=self.named_layer[layer_name])

        self.plan = self.builder.build_serialized_network(self.network, self.config)
        engine = self.runtime.deserialize_cuda_engine(self.plan)

        return engine

    ##########
    # Layers #
    ##########
    def rename_Common(self, source_node):
        logger.warning("PyTorch parser will skip operator [%s] with name [%s]."
              % (source_node.type, source_node.name)) 

        return None

    def rename_Data(self, source_node):
        tensor = self.network.add_input(source_node.name, dtype=trt.float32, shape=source_node.output_shape)

        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name
        caffe_layer.type = 'Input'    
        caffe_layer.top.append(source_node.name)

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = tensor
        tensor.name = source_node.name
       
        return tensor

    def rename_Conv(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None      

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

        if 'group' not in attr:
            num_groups = 1
        else:
            num_groups = attr['group']

        if 'dilations' not in attr:
            dilations = [1, 1]
        else:
            dilations = attr['dilations']

        if source_node.weights_name == "":
            bias_name = 'bias'
            weights_name = 'weight'
        else:
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
            bias = trt.Weights()

        layer = self.network.add_convolution(input=self.named_layer[source_node.in_edges[0]], num_output_maps=num_filter, kernel_shape=kernel_shape, kernel=weight, bias=bias)    
        layer.stride = strides
        layer.padding = pads
        layer.num_groups = num_groups
        layer.dilation = dilations
        layer.name = source_node.name

        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name
        caffe_layer.type = 'Convolution'       
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)

        return layer

    def rename_Relu(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None   

        layer = self.network.add_activation(self.named_layer[source_node.in_edges[0]], type=trt.ActivationType.RELU)
        layer.name = source_node.name
        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name
        caffe_layer.type = 'ReLU'     
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)   

        return layer

    def rename_MaxPool(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None
        
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

        layer = self.network.add_pooling(self.named_layer[source_node.in_edges[0]], trt.PoolingType.MAX, window_size=kernel_shape)
        layer.stride = strides
        layer.padding = pads
        if 'ceil_mode' in attr:
            if attr['ceil_mode'] == 1:
                layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

        layer.name = source_node.name
        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name
        caffe_layer.type = 'Pooling'       
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)

        return layer

    def rename_Add(self, source_node):
        if not self.is_main(source_node.in_edges[0:2]):
            return None

        layer = self.network.add_elementwise(self.named_layer[source_node.in_edges[0]], self.named_layer[source_node.in_edges[1]], trt.ElementWiseOperation.SUM)
        layer.name = source_node.name 
        
        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name     
        caffe_layer.type = 'Eltwise'     
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])
        caffe_layer.bottom.append(source_node.in_edges[1])

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)   

        return layer

    def rename_GlobalAveragePool(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None   
        
        kernel_shape  = self.named_layer[source_node.in_edges[0]].shape
        layer = self.network.add_pooling(self.named_layer[source_node.in_edges[0]], trt.PoolingType.AVERAGE, window_size=kernel_shape[2:])
        layer.name = source_node.name
        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name   
        caffe_layer.type = 'Pooling'       
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)

        return layer

    def rename_Flatten(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None

        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name  
        caffe_layer.type = 'Flatten'        
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = self.named_layer[source_node.in_edges[0]]

        return

    def rename_FullyConnected(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None
        
        if source_node.weights_name == "":
            bias_name = 'bias'
            weights_name = 'weight'
        else:
            bias_name = '{0}.bias'.format(source_node.weights_name)
            weights_name = '{0}.weight'.format(source_node.weights_name)

        W = self.state_dict[weights_name].numpy().transpose()

        _, output_channels = W.shape

        weight = self.state_dict[weights_name].numpy()

        self.set_weight(source_node.name, 'weights', W )

        if bias_name in self.state_dict:
            bias = self.state_dict[bias_name].numpy()
        else:
            bias = trt.Weights()
        
        if(len(self.named_layer[source_node.in_edges[0]].shape) == 4):
            layer = self.network.add_fully_connected(input=self.named_layer[source_node.in_edges[0]], num_outputs=output_channels, kernel=weight, bias=bias)
        else:
            shuffle_layer = self.network.add_shuffle(self.named_layer[source_node.in_edges[0]])
            shuffle_layer.reshape_dims = tuple(self.named_layer[source_node.in_edges[0]].shape) + (1, 1)

            fc_layer = self.network.add_fully_connected(input=shuffle_layer.get_output(0), num_outputs=output_channels, kernel=weight, bias=bias)
        
            layer = self.network.add_shuffle(fc_layer.get_output(0))
            layer.reshape_dims = tuple(source_node.output_shape[1:])

        layer.name = source_node.name         
        
        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name   
        caffe_layer.type = 'InnerProduct'       
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)   

        return layer

    def rename_Sigmoid(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None

        layer = self.network.add_activation(self.named_layer[source_node.in_edges[0]], type=trt.ActivationType.SIGMOID)
        layer.name = source_node.real_name

        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name  
        caffe_layer.type = 'Sigmoid'      
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)

        return layer

    def rename_ReduceSum(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None
        
        attr = source_node.attrs
        axes = np.power(2, attr['axes'][0]) # bit wise 
        layer = self.network.add_reduce(self.named_layer[source_node.in_edges[0]], trt.ReduceOperation.SUM, axes=axes, keep_dims=True)
        layer.name = source_node.name
        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name   
        caffe_layer.type = 'Pooling'      
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)   

        return layer

    def rename_Div(self, source_node):
        if not self.is_main(source_node.in_edges[0:2]):
            return None   
        
        layer = self.network.add_elementwise(self.named_layer[source_node.in_edges[0]], self.named_layer[source_node.in_edges[1]], trt.ElementWiseOperation.DIV)
        layer.name = source_node.name
        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name      
        caffe_layer.type = 'Eltwise'   
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)   
        
        return layer

    def rename_Mul(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None
        
        attr = source_node.attrs

        if 'scale' in attr:
            layer = self.network.add_scale_nd(self.named_layer[source_node.in_edges[0]], mode=trt.ScaleMode(0), scale=np.array(attr['scale']).astype(np.float32), channel_axis=1)  
            layer.channel_axis = 3
        else:
            layer = self.network.add_elementwise(self.named_layer[source_node.in_edges[0]], self.named_layer[source_node.in_edges[1]], trt.ElementWiseOperation.PROD)

        layer.name = source_node.name
        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name   
        caffe_layer.type = 'Eltwise'      
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])
        caffe_layer.bottom.append(source_node.in_edges[1])

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)

        return layer

    def rename_ConvTranspose(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None   
        
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

        if source_node.weights_name == "":
            bias_name = 'bias'
            weights_name = 'weight'
        else:
            bias_name = '{0}.bias'.format(source_node.weights_name)
            weights_name = '{0}.weight'.format(source_node.weights_name)

        weight = self.state_dict[weights_name]
        weight = weight.numpy()

        num_output_maps = list(weight.shape)[1]

        # handle bias
        if bias_name in self.state_dict:
            bias = self.state_dict[bias_name].numpy()
        else:
            bias = trt.Weights()

        if 'group' not in attr:
            num_groups = 1
        else:
            num_groups = attr['group']      

        if 'output_padding' not in attr:
            output_padding = [0, 0]
        else:
            output_padding = attr['output_padding']  

        layer = self.network.add_deconvolution(input=self.named_layer[source_node.in_edges[0]], num_output_maps=num_output_maps,
                                        kernel_shape=kernel_shape, kernel=weight, bias=bias)
        layer.stride = strides
        layer.num_groups = num_groups

        if output_padding[0] + output_padding[1] > 0:
            layer.pre_padding = pads
            layer.post_padding = trt.tensorrt.DimsHW(pads[0] - output_padding[0], pads[1] - output_padding[1])
        else:
            layer.padding = pads   

        layer.name = source_node.name
        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name    
        caffe_layer.type = 'Deconvolution'
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)

        return layer

    def rename_Concat(self, source_node):
        if not self.is_main(source_node.in_edges):
            return None

        attr = source_node.attrs
        
        input_ = [self.named_layer[x] for x in source_node.in_edges]
        layer = self.network.add_concatenation(input_)
        if attr['axis'] < 0:
            layer.axis = len(source_node.output_shape) + attr['axis']
        else:
            layer.axis = attr['axis']
        layer.name = source_node.name

        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name   
        caffe_layer.type = 'Concat'     
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.extend(source_node.in_edges)

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)   

        return layer

    def rename_Reshape(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None

        attr = source_node.attrs

        shape = None
        if 'shape' in attr:
            shape = attr['shape']
        elif source_node.output_shape is not None:
            shape = source_node.output_shape
        else:
            raise Exception('Shape get not be retrived')   

        if isinstance(self.named_layer[source_node.in_edges[0]], trt.tensorrt.ITensor):
            input_ = self.named_layer[source_node.in_edges[0]]
        else:
            input_ = self.named_layer[source_node.in_edges[0]]

        layer = self.network.add_shuffle(input_)
        layer.reshape_dims = shape
        layer.name = source_node.name
        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name    
        caffe_layer.type = 'Reshape'    
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)   

        return layer

    def rename_Permute(self, source_node):
        if not self.is_main(source_node.in_edges):
            return None

        attr = source_node.attrs

        if 'perm' in attr:
            order = attr['perm']
        else:
            order = []

        layer = self.network.add_shuffle(self.named_layer[source_node.in_edges[0]])
        layer.first_transpose = order
        layer.name = source_node.name
        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name 
        caffe_layer.type = 'Permute'       
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)   

        return layer

    def rename_Resize(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None
        
        attr = source_node.attrs

        layer = self.network.add_resize(self.named_layer[source_node.in_edges[0]])
        if 'scale_factor' in attr:
            scale = attr['scale_factor'][0]
            layer.scales = [1, 1, scale, scale]
        else:
            input_shape = self.named_layer[source_node.in_edges[0]].shape
            shape = list(input_shape[0:2]) + attr['output_size']
            layer.shape = shape
        
        if attr['mode'] == "linear":
            layer.resize_mode = trt.ResizeMode.LINEAR
        else:
            layer.resize_mode = trt.ResizeMode.NEAREST

        if attr['coordinate_transformation_mode'] == "pytorch_half_pixel":
            layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
        elif attr['coordinate_transformation_mode'] == "asymmetric":
            layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC       
        else:
            raise

        if attr['nearest_mode'] == 'floor':
            layer.nearest_rounding = trt.ResizeRoundMode.FLOOR 
        else:
            raise

        layer.name = source_node.name
        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name 
        caffe_layer.type = 'Upsample'        
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)   

        return layer

    def rename_AveragePool(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None
 
        attr = source_node.attrs

        kernel_shape  = attr['kernel_shape']
        stride = attr['strides']
        layer = self.network.add_pooling(self.named_layer[source_node.in_edges[0]], trt.PoolingType.AVERAGE, window_size=kernel_shape)
        layer.stride = stride
        if "pads" in attr:
            layer.padding = (attr['pads'][0], attr['pads'][2])
        if "count_include_pad" not in attr:
            layer.average_count_excludes_padding = False
        if 'ceil_mode' in attr:
            if attr['ceil_mode'] == 1:
                layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP
       
        layer.name = source_node.name
        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name   
        caffe_layer.type = 'GlobalPooling'       
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)

        return layer

    def rename_Gather(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None
 
        attr = source_node.attrs
        
        axis  = attr['axis']
        indices = attr['indices']
        # dims = trt.Dims(shape=[len(indices)])

        const_layer = self.network.add_constant(shape=[len(indices)], weights=np.array(indices).astype(np.int32))
        layer = self.network.add_gather(self.named_layer[source_node.in_edges[0]], indices=const_layer.get_output(0), axis=axis)
        layer.name = source_node.name
        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name   
        caffe_layer.type = 'Gather'       
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)

        return layer

    def rename_BatchNormalization(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None
 
        if source_node.weights_name == "":
            mean_name = 'running_mean'
            var_name = 'running_var'
        else:
            mean_name = '{0}.running_mean'.format(source_node.weights_name)
            var_name = '{0}.running_var'.format(source_node.weights_name)

        mean = self.state_dict[mean_name].numpy()
        variance = self.state_dict[var_name].numpy()

        if source_node.weights_name == "":
            bias_name = 'bias'
            weights_name = 'weight'
        else:
            bias_name = '{0}.bias'.format(source_node.weights_name)
            weights_name = '{0}.weight'.format(source_node.weights_name)

        scale = self.state_dict[weights_name].numpy()

        if bias_name in self.state_dict:
            bias = self.state_dict[bias_name].numpy()
        else:
            bias = np.zeros_like(mean)
        
        bn_eps = 1e-05
        bn_var_rsqrt = 1 / np.sqrt(variance + bn_eps)

        new_w = (scale * bn_var_rsqrt).reshape([-1] + [1] * (len(scale.shape) - 1))
        new_b = (- mean) * bn_var_rsqrt * scale + bias

        layer = self.network.add_scale(self.named_layer[source_node.in_edges[0]], mode=trt.ScaleMode(1), shift=new_b, scale=new_w)
        layer.channel_axis = 1
        layer.name = source_node.name
        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name   
        caffe_layer.type = 'BatchNorm'       
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)

        return layer     

    def rename_Softmax(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None
 
        attr = source_node.attrs

        layer = self.network.add_softmax(self.named_layer[source_node.in_edges[0]])
        layer.axes  = (1 << (attr['axis']))
        layer.name = source_node.name
        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name   
        caffe_layer.type = 'Softmax'       
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])
 
        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)

        return layer

    def rename_Clip(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None
 
        attr = source_node.attrs

        layer = self.network.add_activation(self.named_layer[source_node.in_edges[0]], type=trt.ActivationType.CLIP)
        layer.alpha = attr['min']
        layer.beta = attr['max']
        layer.name = source_node.name

        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name   
        caffe_layer.type = 'Relu6'       
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])
 
        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)

        return layer

    def rename_HardSigmoid(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None
 
        attr = source_node.attrs

        layer = self.network.add_activation(self.named_layer[source_node.in_edges[0]], type=trt.ActivationType.HARD_SIGMOID)
        layer.alpha = attr['alpha']
        layer.beta = 0.5
        layer.name = source_node.name

        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name   
        caffe_layer.type = 'HardSigmoid'       
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])
 
        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)

        return layer

    def rename_HardSwish(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None
 
        attr = source_node.attrs

        layer_hardsigmoid = self.network.add_activation(self.named_layer[source_node.in_edges[0]], type=trt.ActivationType.HARD_SIGMOID)
        layer_hardsigmoid.alpha = 0.5
        layer_hardsigmoid.beta = 0.5

        layer = self.network.add_elementwise(self.named_layer[source_node.in_edges[0]], layer_hardsigmoid.get_output(0), trt.ElementWiseOperation.PROD)        
        layer.name = source_node.name

        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name   
        caffe_layer.type = 'HardSwish'       
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])
 
        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)

        return layer

    def rename_Pad(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None
        
        attr = source_node.attrs

        pre_padding = (0, 0)
        post_padding = (0, 0)
        if 'pads' in attr:
        # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
            pre_padding = (attr['pads'][2], attr['pads'][3]) # top left
            post_padding = (attr['pads'][6], attr['pads'][7]) # bottom right

        attr = source_node.attrs

        layer = self.network.add_padding(self.named_layer[source_node.in_edges[0]], pre_padding, post_padding)        
        layer.name = source_node.name

        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name   
        caffe_layer.type = 'Pad'       
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])
 
        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)

        return layer           

    def rename_Split(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None
 
        attr = source_node.attrs

        input_shape = self.named_layer[source_node.in_edges[0]].shape
        start = [0, 0, 0, 0]
        for idx, output_id in enumerate(source_node.output_ids):
            shape = input_shape
            shape[attr['axis']] = attr['split'][idx]
            stride = [input_shape[0], 1, 1, 1]            
            layer = self.network.add_slice(self.named_layer[source_node.in_edges[0]], start, shape, stride)        
            output_name = source_node.name + ':' + output_id
            self.named_layer[output_name] = layer.get_output(0)
            layer.name = output_name
            start[attr['axis']] = start[attr['axis']] + attr['split'][idx]

        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name   
        caffe_layer.type = 'Split'       
        for output_id in source_node.output_ids:
            output_id = source_node.name + ':' + output_id
            caffe_layer.top.append(output_id)
        caffe_layer.bottom.append(source_node.in_edges[0])
 
        self.main_layers.append(caffe_layer)

        return layer

    def rename_ReduceMean(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None
 
        attr = source_node.attrs
        axes = 0
        for d in attr['axes']:
            axes |= 1 << (d)  # -1 to remove batch dimension

        layer = self.network.add_reduce(self.named_layer[source_node.in_edges[0]], op=trt.ReduceOperation.AVG, axes=axes, keep_dims=attr['keepdims'])
        layer.name = source_node.name

        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name   
        caffe_layer.type = 'ReduceMean'       
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])
 
        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)

        return layer

    def rename_LeakyRelu(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None   

        attr = source_node.attrs

        layer = self.network.add_activation(self.named_layer[source_node.in_edges[0]], type=trt.ActivationType.LEAKY_RELU)
        layer.alpha = attr['alpha']
        layer.name = source_node.name
        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name
        caffe_layer.type = 'LeakyReLU'     
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)   

        return layer

    def rename_Upsample(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None
        
        attr = source_node.attrs

        layer = self.network.add_resize(self.named_layer[source_node.in_edges[0]])
        if 'scale_factor' in attr:
            scale = attr['scale_factor'][0]
            layer.scales = [1, 1, scale, scale]
        else:
            input_shape = self.named_layer[source_node.in_edges[0]].shape
            shape = list(input_shape[0:2]) + attr['output_size']
            layer.shape = shape
        
        if attr['mode'] == "linear":
            layer.resize_mode = trt.ResizeMode.LINEAR
        else:
            layer.resize_mode = trt.ResizeMode.NEAREST

        if attr['coordinate_transformation_mode'] == "pytorch_half_pixel":
            layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
        elif attr['coordinate_transformation_mode'] == "asymmetric":
            layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC       
        else:
            raise

        if attr['nearest_mode'] == 'floor':
            layer.nearest_rounding = trt.ResizeRoundMode.FLOOR 
        else:
            raise

        layer.name = source_node.name
        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name 
        caffe_layer.type = 'Upsample'        
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)   

        return layer

    def rename_ArgMax(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None   

        attr = source_node.attrs
        axes = 1 << (attr['axis'])
        layer = self.network.add_topk(self.named_layer[source_node.in_edges[0]], op=trt.TopKOperation.MAX, k=1, axes=axes)
        layer.name = source_node.name
        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name
        caffe_layer.type = 'Argmax'     
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(1)   

        return layer

    def rename_BilinearInterpolate(self, source_node):
        if not self.is_main(source_node.in_edges[0:1]):
            return None
        
        attr = source_node.attrs
        logger.info(attr)
        layer = self.network.add_resize(self.named_layer[source_node.in_edges[0]])
        if 'scale' in attr:
            scale = attr['scale'][0]
            layer.scales = [1, 1, scale, scale]
        else:
            input_shape = self.named_layer[source_node.in_edges[0]].shape
            shape = list(input_shape[0:2]) + attr['size']
            layer.shape = shape
        
        if attr['mode'] == "linear":
            layer.resize_mode = trt.ResizeMode.LINEAR
        else:
            layer.resize_mode = trt.ResizeMode.NEAREST

        if attr['align_corners'] == 1:
            layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ALIGN_CORNERS

        layer.name = source_node.name
        caffe_layer = pb2.LayerParameter()
        caffe_layer.name = source_node.name 
        caffe_layer.type = 'BilinearInterpolate'        
        caffe_layer.top.append(source_node.name)
        caffe_layer.bottom.append(source_node.in_edges[0])

        self.main_layers.append(caffe_layer)
        self.named_layer[source_node.name] = layer.get_output(0)   

        return layer        