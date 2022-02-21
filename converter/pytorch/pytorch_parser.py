#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import numpy as np
from converter.core.parser import Parser
from converter.pytorch.pytorch_graph import PytorchGraph
import caffe.proto.caffe_pb2 as pb2

import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

def as_blob(array):
    blob = pb2.BlobProto()
    blob.shape.dim.extend(array.shape)
    blob.data.extend(array.astype(float).flat)
    return blob

class PytorchParser(Parser):
    layer_map = {
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
}

    @property
    def src_graph(self):
        return self.pytorch_graph


    def __init__(self, model, input_shape, opset_version, fuse=False):
        super(PytorchParser, self).__init__()
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
        self.caffe_net = []
        self.bottoms = list()
        self.main_layers = []

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
        if isinstance(self.input_shape, tuple):
            for idx, shape in enumerate(self.input_shape):
                name = "data_" + str(idx)
                func = getattr(self, "rename_Data")
                layer_data = func(shape, name)
                self.caffe_net.append(layer_data)
                self.bottoms.append(name)
        else:                                    
            func = getattr(self, "rename_Data")
            layer_data = func(self.input_shape, "data")
            self.caffe_net.append(layer_data)  
            self.bottoms.append("data") 

        for node in self.src_graph.topological_sort:
            current_node = self.src_graph.get_node(node)
            self.named_node[current_node.real_name] = current_node
            onnx_node_type = current_node.type
            node_type = PytorchParser.layer_map[onnx_node_type]

            if hasattr(self, "rename_" + node_type):
                func = getattr(self, "rename_" + node_type)
                layer_data = func(current_node)
                if layer_data == None:
                    continue
                elif(isinstance(layer_data, tuple)):
                    self.caffe_net.append(layer_data[0])
                    self.caffe_net.append(layer_data[1])
                    self.named_layer[layer_data[0].name] = layer_data[0]
                    self.named_layer[layer_data[1].name] = layer_data[1] # some batchnorm will not be eliminated
                    # self.named_layer[layer_data[0].name.rsplit('_', 1)[0]] = layer_data[1]
                else:
                    self.caffe_net.append(layer_data)                    
                    self.named_layer[layer_data.name] = layer_data
            else:
                self.rename_Common(current_node)

        text_net = pb2.NetParameter()
        binary_weights = pb2.NetParameter()
        binary_weights.CopyFrom(text_net)

        if self.fuse:
            for layer in self.caffe_net:
                if layer.type in ["ReLU"] and self.named_layer[layer.bottom[0]].type == "Convolution":
                    self.named_layer[layer.bottom[0]].top[0] = layer.top[0]                
                    layer.bottom[0] = layer.top[0]

        for layer in self.main_layers:
            binary_weights.layer.extend([layer])
            layer_proto = pb2.LayerParameter()
            layer_proto.CopyFrom(layer)
            del layer_proto.blobs[:]
            text_net.layer.extend([layer_proto])

        return text_net, binary_weights

    ##########
    # Layers #
    ##########
    def rename_Common(self, source_node):
        print("PyTorch parser will skip operator [%s] with name [%s]."
              % (source_node.type, source_node.name)) 

        return None

    def rename_Data(self, shape, name):
        layer = pb2.LayerParameter()
        layer.type = 'Input'
        input_shape = pb2.BlobShape()
        input_shape.dim.extend(shape)
        layer.input_param.shape.extend([input_shape])
        layer.top.append(name)
        layer.name = name
        self.main_layers.append(layer)
        self.named_layer[name] = layer
        return layer

    def rename_Conv(self, source_node):
        attr = source_node.attrs
        kwargs = dict()
        layer = pb2.LayerParameter()

        layer.type = "Convolution"
        # dilation
        if 'dilations' in attr:
            kwargs['dilations'] = [1] + attr['dilations'] + [1]
            layer.convolution_param.dilation.extend([attr['dilations'][0]])
        else:
            kwargs['dilations'] = [1] + [1, 1] + [1]
            layer.convolution_param.dilation.extend(1)

        if len(attr['pads']) == 4:
            kwargs['pads'] = [0] + attr['pads'][0:2] + [0, 0] + attr['pads'][2:] + [0]
            if attr['pads'][0] == attr['pads'][1]:
                layer.convolution_param.pad.extend([attr['pads'][0]])
            else:
                layer.convolution_param.pad_h = attr['pads'][0]
                layer.convolution_param.pad_w = attr['pads'][1]
        elif len(attr['pads']) == 2:
            kwargs['pads'] = ( [0] + attr['pads'][0:2] + [0] ) *2
            if attr['pads'][0] == attr['pads'][1]:
                layer.convolution_param.pad.extend([attr['pads'][0]])
            else:
                layer.convolution_param.pad_h = attr['pads'][0]
                layer.convolution_param.pad_w = attr['pads'][1]

        if 'strides' not in attr:
            kwargs['strides'] = [1] + [1, 1] + [1]
        else:
            kwargs['strides'] = [1] + attr['strides'] + [1]
            if attr['strides'][0] == attr['strides'][1]:
                layer.convolution_param.stride.extend([attr['strides'][0]])
            else:
                layer.convolution_param.stride_h = attr['strides'][0]
                layer.convolution_param.stride_w = attr['strides'][1]

        if 'kernel_shape' not in attr:
            kwargs['kernel_shape'] = [1] + [1, 1] + [1]
            layer.convolution_param.kernel_size.extend([1])
        else:
            kwargs['kernel_shape'] = [1] + attr['kernel_shape'] + [1]
            if attr['kernel_shape'][0] == attr['kernel_shape'][1]:
                layer.convolution_param.kernel_size.extend([attr['kernel_shape'][0]])
            else:
                layer.convolution_param.kernel_h = attr['kernel_shape'][0]
                layer.convolution_param.kernel_w = attr['kernel_shape'][1]

        kwargs['group'] = attr['group']
        layer.convolution_param.group = attr['group']

        bias_name = '{0}.bias'.format(source_node.weights_name)
        weights_name = '{0}.weight'.format(source_node.weights_name)
        weight = self.state_dict[weights_name]

        weight = weight.numpy()

        self.set_weight(source_node.name, 'weights', weight)
        kwargs['kernel_shape'] = list(weight.shape)
        layer.convolution_param.num_output = list(weight.shape)[0]

        # handle bias
        if bias_name in self.state_dict:
            bias = self.state_dict[bias_name].numpy()
            self.set_weight(source_node.name, 'bias', bias)
            kwargs['use_bias'] = True
            layer.convolution_param.bias_term = True
            layer.blobs.extend([as_blob(weight), as_blob(bias)])
        else:
            kwargs['use_bias'] = False
            layer.convolution_param.bias_term = False
            layer.blobs.extend([as_blob(weight)])

        for b in source_node.in_edges:
            layer.bottom.append(b)

        if len(source_node.in_edges) == 0:
            layer.bottom.append(self.bottoms.pop(0))

        layer.top.append(source_node.name)

        layer.name = source_node.real_name

        if self.is_main(layer.bottom):
            self.main_layers.append(layer)
        return layer

    def rename_PRelu(self, source_node):
        attr = source_node.attrs
        kwargs = dict()
        layer = pb2.LayerParameter()
        layer.type = "PReLU"

        bias_name = '{0}.bias'.format(source_node.weights_name)
        weights_name = '{0}.weight'.format(source_node.weights_name)

        weight = self.state_dict[weights_name]

        weight = weight.numpy()
        dim = weight.ndim

        layer.prelu_param.channel_shared = True if dim == 1 else False
        layer.blobs.extend([as_blob(weight[0])])

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)

        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)
        return layer

    def rename_GlobalAveragePool(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.AVE
        layer.pooling_param.global_pooling = True
        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)
        return layer

    def rename_Sigmoid(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Sigmoid"

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)
        return layer

    def rename_BatchNormalization(self, source_node):
        attr = source_node.attrs

        layer_bn = pb2.LayerParameter()
        layer_bn.type = "BatchNorm"

        layer_bn.batch_norm_param.use_global_stats = 1
        layer_bn.batch_norm_param.eps = attr['epsilon']

        mean_name = '{0}.running_mean'.format(source_node.weights_name)
        var_name = '{0}.running_var'.format(source_node.weights_name)

        mean = self.state_dict[mean_name].numpy()
        variance = self.state_dict[var_name].numpy()

        layer_bn.blobs.extend([as_blob(mean), as_blob(variance), as_blob(np.array([1.]))])

        for b in source_node.in_edges:
            layer_bn.bottom.append(b)

        layer_bn.top.append(source_node.real_name + '_bn')
        layer_bn.name = source_node.real_name + '_bn'

        layer_scale = pb2.LayerParameter()
        layer_scale.type = "Scale"

        bias_name = '{0}.bias'.format(source_node.weights_name)
        weights_name = '{0}.weight'.format(source_node.weights_name)

        weight = self.state_dict[weights_name].numpy()

        if bias_name in self.state_dict:
            bias = self.state_dict[bias_name].numpy()
            layer_scale.scale_param.bias_term = True
            layer_scale.blobs.extend([as_blob(weight), as_blob(bias)])
        else:
            layer_scale.scale_param.bias_term = False
            layer_scale.blobs.extend([as_blob(weight)])

        layer_scale.bottom.append(source_node.real_name + '_bn')

        layer_scale.top.append(source_node.name)

        layer_scale.name = source_node.real_name
        if self.is_main(layer_bn.bottom):
            self.main_layers.append(layer_bn)
            self.main_layers.append(layer_scale)
        return layer_bn, layer_scale

    def rename_Relu(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "ReLU"

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)
        return layer

    def rename_MaxPool(self, source_node):
        attr = source_node.attrs      
        kwargs = dict()

        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.MAX

        if 'pads' in attr:
            if len(attr['pads']) == 4:
                kwargs['pads'] = [0] + attr['pads'][0:2] + [0, 0] + attr['pads'][2:] + [0]
                if attr['pads'][0] == attr['pads'][1] and attr['pads'][2] == attr['pads'][3]:
                    layer.pooling_param.pad = attr['pads'][0]
                else:
                    layer.pooling_param.pad_h = attr['pads'][0]
                    layer.pooling_param.pad_w = attr['pads'][1]
            elif len(attr['pads']) == 2:
                kwargs['pads'] = ([0] + attr['pads'][0:2] + [0]) * 2
                if attr['pads'][0] == attr['pads'][1]:
                    layer.pooling_param.pad = attr['pads'][0]
                else:
                    layer.pooling_param.pad_h = attr['pads'][0]
                    layer.pooling_param.pad_w = attr['pads'][1]

        if 'dilations' not in attr:
            kwargs['strides'] = [1] + [1, 1] + [1]
            layer.pooling_param.stride = 1
        else:
            kwargs['strides'] = [1] + attr['strides'] + [1]
            if attr['strides'][0] == attr['strides'][1]:
                layer.pooling_param.stride = attr['strides'][0]
            else:
                layer.pooling_param.stride_h = attr['strides'][0]
                layer.pooling_param.stride_w = attr['strides'][1]

        if 'strides' not in attr:
            kwargs['strides'] = [1] + [1, 1] + [1]
            layer.pooling_param.stride = 1
        else:
            kwargs['strides'] = [1] + attr['strides'] + [1]
            if attr['strides'][0] == attr['strides'][1]:
                layer.pooling_param.stride = attr['strides'][0]
            else:
                layer.pooling_param.stride_h = attr['strides'][0]
                layer.pooling_param.stride_w = attr['strides'][1]

        if 'kernel_shape' not in attr:
            kwargs['kernel_shape'] = [1] + [1, 1] + [1]
            layer.pooling_param.kernel_size.extend(1)
        else:
            kwargs['kernel_shape'] = [1] + attr['kernel_shape'] + [1]
            if attr['kernel_shape'][0] == attr['kernel_shape'][1]:
                layer.pooling_param.kernel_size = attr['kernel_shape'][0]
            else:
                layer.pooling_param.kernel_h = attr['kernel_shape'][0]
                layer.pooling_param.kernel_w = attr['kernel_shape'][1]

        if 'ceil_mode' not in attr:
            layer.pooling_param.ceil_mode = 0
        else:
            layer.pooling_param.ceil_mode = attr['ceil_mode']

        for b in source_node.in_edges:
            layer.bottom.append(b)
        
        if len(source_node.output_ids) > 1:
            for output_id in source_node.output_ids:
                output_id = source_node.name + ':' + output_id
                layer.top.append(output_id)
        else:
            layer.top.append(source_node.name)

        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)
        return layer

    def rename_Add(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Eltwise"

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)
        return layer

    def rename_AveragePool(self, source_node):
        attr = source_node.attrs
        kwargs = dict()
        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.AVE

        if 'pads' not in attr:
            layer.pooling_param.pad = 0
        else:
            if len(attr['pads']) == 4:
                kwargs['pads'] = [0] + attr['pads'][0:2] + [0, 0] + attr['pads'][2:] + [0]
                if attr['pads'][0] == attr['pads'][1]:
                    layer.pooling_param.pad = attr['pads'][0]
                else:
                    layer.pooling_param.pad_h = attr['pads'][0]
                    layer.pooling_param.pad_w = attr['pads'][1]
            elif len(attr['pads']) == 2:
                kwargs['pads'] = ([0] + attr['pads'][0:2] + [0]) * 2
                if attr['pads'][0] == attr['pads'][1]:
                    layer.pooling_param.pad = attr['pads'][0]
                else:
                    layer.pooling_param.pad_h = attr['pads'][0]
                    layer.pooling_param.pad_w = attr['pads'][1]

        if 'strides' not in attr:
            kwargs['strides'] = [1] + [1, 1] + [1]
            layer.pooling_param.stride = 1
        else:
            kwargs['strides'] = [1] + attr['strides'] + [1]
            if attr['strides'][0] == attr['strides'][1]:
                layer.pooling_param.stride = attr['strides'][0]
            else:
                layer.pooling_param.stride_h = attr['strides'][0]
                layer.pooling_param.stride_w = attr['strides'][1]

        if 'kernel_shape' not in attr:
            kwargs['kernel_shape'] = [1] + [1, 1] + [1]
            layer.pooling_param.kernel_size.extend(1)
        else:
            kwargs['kernel_shape'] = [1] + attr['kernel_shape'] + [1]
            if attr['kernel_shape'][0] == attr['kernel_shape'][1]:
                layer.pooling_param.kernel_size = attr['kernel_shape'][0]
            else:
                layer.pooling_param.kernel_h = attr['kernel_shape'][0]
                layer.pooling_param.kernel_w = attr['kernel_shape'][1]

        if 'ceil_mode' not in attr:
            kwargs['ceil_mode'] = 0
        else:
            if attr['ceil_mode'] != 1:
                layer.pooling_param.stride_h = attr['strides'][0]
                layer.pooling_param.stride_w = attr['strides'][1]

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)
        return layer

    def rename_Flatten(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        layer.type = "Flatten"

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)
        return layer

    def rename_FullyConnected(self, source_node):
        attr = source_node.attrs

        layer = pb2.LayerParameter()
        layer.type = "InnerProduct"

        bias_name = '{0}.bias'.format(source_node.weights_name)
        weights_name = '{0}.weight'.format(source_node.weights_name)

        W = self.state_dict[weights_name].numpy().transpose()

        input_channels, output_channels = W.shape

        weight = self.state_dict[weights_name].numpy()

        # weights
        self.set_weight(source_node.name, 'weights', W )

        # use_bias
        if bias_name in self.state_dict:
            bias = self.state_dict[bias_name].numpy()
            layer.inner_product_param.bias_term = True
            layer.blobs.extend([as_blob(weight), as_blob(bias)])
        else:
            layer.inner_product_param.bias_term = False
            layer.blobs.extend([as_blob(weight)])

        layer.inner_product_param.num_output = output_channels

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)
        return layer

    def rename_Dropout(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        layer.type = "Dropout"
        layer.dropout_param.dropout_ratio = attr['ratio']

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name

        return layer

    def rename_Softmax(self, source_node):
        attr = source_node.attrs

        layer = pb2.LayerParameter()
        layer.type = 'Softmax'
        layer.softmax_param.axis = attr['axis']

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)
        return layer

    def rename_Permute(self, source_node):
        attr = source_node.attrs
        kwargs = dict()
        layer = pb2.LayerParameter()
        layer.type = "Permute"

        if len(attr['perm']) == 4:
            layer.permute_param.order.extend([attr['perm'][0]])
            layer.permute_param.order.extend([attr['perm'][1]])
            layer.permute_param.order.extend([attr['perm'][2]])
            layer.permute_param.order.extend([attr['perm'][3]])

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)

        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)
        return layer

    def rename_Upsample(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        layer.type = "Upsample"

        if 'scale_factor' in attr:
            layer.upsample_param.scale = attr['scale_factor'][0]

        layer.bottom.append(source_node.in_edges[0])

        layer.top.append(source_node.name)

        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)        
        return layer

    def rename_Concat(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        layer.type = "Concat"
        layer.concat_param.axis = attr['axis']

        for b in source_node.in_edges:
            layer.bottom.append(b)           

        layer.top.append(source_node.name)

        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)   

        return layer

    def rename_Unsqueeze(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        layer.type = "Unsqueeze"

        if 'axes' in attr:
            layer.unsqueeze_param.dim = attr['axes'][0]
        else:
            layer.unsqueeze_param.dim = 0            

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)

        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)        
        return layer

    def rename_Relu6(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        layer.type = "ReLU6"

        if 'max' in attr:
            layer.relu6_param.threshold = attr['max']

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)

        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)        
        return layer     

    def rename_Pad(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        layer.type = "Pad"

        if 'mode' in attr:
            mode = attr['mode']
            if mode == "reflect":
                layer.pad_param.pad_type = pb2.PadParameter.REFLECT

        if 'pads' in attr:
        # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
            layer.pad_param.pad_u = attr['pads'][2]
            layer.pad_param.pad_d = attr['pads'][6]
            layer.pad_param.pad_l = attr['pads'][3]
            layer.pad_param.pad_r = attr['pads'][7]

        if self.opset_version > 9:
            if len(source_node.in_edges) == 1:
                layer.bottom.append(self.bottoms.pop(0))
            else:
                layer.bottom.append(source_node.in_edges[0])   
        else:
            if len(source_node.in_edges) == 0:
                layer.bottom.append(self.bottoms.pop(0))
            else:
                layer.bottom.append(source_node.in_edges[0])
        
        layer.top.append(source_node.name)

        layer.name = source_node.real_name

        if self.is_main(layer.bottom):
            self.main_layers.append(layer)

        return layer   

    def rename_HardSwish(self, source_node):
        attr = source_node.attrs

        layer = pb2.LayerParameter()
        layer.type = "HardSwish"

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)

        layer.name = source_node.real_name

        if self.is_main(layer.bottom):
            self.main_layers.append(layer)

        return layer      

    def rename_HardSigmoid(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        layer.type = "HardSigmoid"

        if 'alpha' in attr:
            layer.hardsigmoid_param.alpha = attr['alpha']
        if 'beta' in attr:
            layer.hardsigmoid_param.beta = attr['beta']

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)

        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)

        return layer            

    def rename_Mul(self, source_node):
        if self.named_node[source_node.in_edges[0]].output_shape == None or self.named_node[source_node.in_edges[1]].output_shape == None:
            layer = pb2.LayerParameter()
            layer.type = "Scale"
            layer.scale_param.axis = 0
            layer.bottom.append(source_node.in_edges[0])
            layer.bottom.append(source_node.in_edges[1])
            layer.top.append(source_node.name)
            layer.name = source_node.real_name     
            if self.is_main(layer.bottom):
                self.main_layers.append(layer)
            return layer     
        elif self.named_node[source_node.in_edges[0]].output_shape[-2:] == self.named_node[source_node.in_edges[1]].output_shape[-2:]:
            layer = pb2.LayerParameter()
            layer.type = "Scale"
            layer.scale_param.axis = 0
            layer.bottom.append(source_node.in_edges[0])
            layer.bottom.append(source_node.in_edges[1])
            layer.top.append(source_node.name)
            layer.name = source_node.real_name
            if self.is_main(layer.bottom):
                self.main_layers.append(layer)    
            return layer

        elif self.named_node[source_node.in_edges[0]].output_shape[-2:] == [1, 1]:
            layer_flatten = pb2.LayerParameter()
            layer_flatten.type = "Flatten"
            layer_flatten.flatten_param.axis = 1
            layer_flatten.bottom.append(source_node.in_edges[0])
            layer_flatten.top.append(source_node.name + '_flatten')
            layer_flatten.name = source_node.real_name + '_flatten'

            layer_scale = pb2.LayerParameter()
            layer_scale.type = "Scale"
            layer_scale.scale_param.axis = 0
            layer_scale.bottom.append(source_node.in_edges[1])            
            layer_scale.bottom.append(source_node.name + '_flatten')
            layer_scale.top.append(source_node.name)
            layer_scale.name = source_node.real_name

            if self.is_main(source_node.in_edges):
                self.main_layers.append(layer_flatten)
                self.main_layers.append(layer_scale)
            return layer_flatten, layer_scale

        elif self.named_node[source_node.in_edges[1]].output_shape[-2:] == [1, 1]:
            layer_flatten = pb2.LayerParameter()
            layer_flatten.type = "Flatten"
            layer_flatten.bottom.append(source_node.in_edges[1])
            layer_flatten.top.append(source_node.name + '_flatten')
            layer_flatten.name = source_node.real_name + '_flatten'

            layer_scale = pb2.LayerParameter()
            layer_scale.type = "Scale"
            layer_scale.scale_param.axis = 0
            layer_scale.bottom.append(source_node.in_edges[0])            
            layer_scale.bottom.append(source_node.name + '_flatten')
            layer_scale.top.append(source_node.name)
            layer_scale.name = source_node.real_name

            if self.is_main(source_node.in_edges):
                self.main_layers.append(layer_flatten)
                self.main_layers.append(layer_scale)
            return layer_flatten, layer_scale      
        else:
            layer = pb2.LayerParameter()
            layer.type = "Scale"
            layer.scale_param.axis = 0
            layer.bottom.append(source_node.in_edges[0])
            layer.bottom.append(source_node.in_edges[1])
            layer.top.append(source_node.name)
            layer.name = source_node.real_name     
            if self.is_main(layer.bottom):
                self.main_layers.append(layer)
            return layer                

    def rename_Slice(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        layer.type = "Slice"

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)
        return layer        

    def rename_Reshape(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        layer.type = "Reshape"

        if 'shape' in attr:
            for each in attr['shape']:
                layer.reshape_param.shape.dim.extend([each])
        elif source_node.output_shape is not None:
             for each in source_node.output_shape:
                 layer.reshape_param.shape.dim.extend([each])
        else:
            raise Exception('Shape get not be retrived')         

        layer.bottom.append(source_node.in_edges[0]) # one input

        layer.top.append(source_node.name)

        layer.name = source_node.real_name

        if self.is_main(layer.bottom):
            self.main_layers.append(layer)        

        return layer        

    def rename_Split(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        layer.type = "Slice"

        layer.slice_param.axis = attr['axis']
        sum_ = 0
        for idx in range(len(attr['split']) - 1):
            sum_ = sum_ + attr['split'][idx]
            layer.slice_param.slice_point.extend([sum_])

        for b in source_node.in_edges:
            layer.bottom.append(b)     

        for output_id in source_node.output_ids:
            output_id = source_node.name + ':' + output_id
            layer.top.append(output_id)

        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)        
        return layer                          

    def rename_LpNormalization(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Normalize"

        layer.norm_param.across_spatial = False
        layer.norm_param.scale_filler.type = "constant"
        layer.norm_param.scale_filler.value = 20
        layer.norm_param.channel_shared = False

        weights_name = '{0}.weight'.format(source_node.weights_name)
        weight = self.state_dict[weights_name]
        weight = weight.numpy().squeeze()
        layer.blobs.extend([as_blob(weight)])

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)

        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)
        return layer             

    def rename_Resize(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        layer.type = "Upsample"

        if 'scale_factor' in attr:
            layer.upsample_param.scale = attr['scale_factor'][0]

        layer.bottom.append(source_node.in_edges[0])

        layer.top.append(source_node.name)

        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)        
        return layer

    def rename_LeakyRelu(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        layer.type = "ReLU"
        layer.relu_param.negative_slope = attr['alpha']

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)
        return layer

    def rename_ReduceMean(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.AVE
        layer.pooling_param.global_pooling = True
        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)
        return layer      

    def rename_BilinearInterpolate(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        if self.opset_version == 9:
            layer.type = "BilinearInterpolate"
            layer.bilinear_interpolate_param.align_corners = attr['align_corners']
            layer.bilinear_interpolate_param.scale_factor = attr['scale'][0]
            layer.bottom.append(source_node.in_edges[0])

            layer.top.append(source_node.name)
            layer.name = source_node.real_name
            if self.is_main(layer.bottom):
                self.main_layers.append(layer)
            return layer  
        else:
            raise Exception('Unsupported opset_version: {}'.format(self.opset_versionc))

    def rename_MaxUnPool(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()

        layer.type = "MaxUnPool"
        layer.max_unpool_param.dst_h = source_node.output_shape[2]
        layer.max_unpool_param.dst_w = source_node.output_shape[3]

        layer.bottom.append(source_node.in_edges[0])
        layer.bottom.append(source_node.in_edges[1])

        layer.top.append(source_node.name)
        layer.name = source_node.real_name

        if self.is_main(layer.bottom):
            self.main_layers.append(layer) 
              
        return layer  
     
    def rename_ConvTranspose(self, source_node):
        attr = source_node.attrs
        kwargs = dict()
        layer = pb2.LayerParameter()

        layer.type = "Deconvolution"
        # dilation
        if 'dilations' in attr:
            kwargs['dilations'] = [1] + attr['dilations'] + [1]
            layer.convolution_param.dilation.extend([attr['dilations'][0]])
        else:
            kwargs['dilations'] = [1] + [1, 1] + [1]
            layer.convolution_param.dilation.extend(1)

        if len(attr['pads']) == 4:
            kwargs['pads'] = [0] + attr['pads'][0:2] + [0, 0] + attr['pads'][2:] + [0]
            if attr['pads'][0] == attr['pads'][1]:
                layer.convolution_param.pad.extend([attr['pads'][0]])
            else:
                layer.convolution_param.pad_h = attr['pads'][0]
                layer.convolution_param.pad_w = attr['pads'][1]
        elif len(attr['pads']) == 2:
            kwargs['pads'] = ( [0] + attr['pads'][0:2] + [0] ) *2
            if attr['pads'][0] == attr['pads'][1]:
                layer.convolution_param.pad.extend([attr['pads'][0]])
            else:
                layer.convolution_param.pad_h = attr['pads'][0]
                layer.convolution_param.pad_w = attr['pads'][1]

        if 'strides' not in attr:
            kwargs['strides'] = [1] + [1, 1] + [1]
        else:
            kwargs['strides'] = [1] + attr['strides'] + [1]
            if attr['strides'][0] == attr['strides'][1]:
                layer.convolution_param.stride.extend([attr['strides'][0]])
            else:
                layer.convolution_param.stride_h = attr['strides'][0]
                layer.convolution_param.stride_w = attr['strides'][1]

        if 'kernel_shape' not in attr:
            kwargs['kernel_shape'] = [1] + [1, 1] + [1]
            layer.convolution_param.kernel_size.extend([1])
        else:
            kwargs['kernel_shape'] = [1] + attr['kernel_shape'] + [1]
            if attr['kernel_shape'][0] == attr['kernel_shape'][1]:
                layer.convolution_param.kernel_size.extend([attr['kernel_shape'][0]])
            else:
                layer.convolution_param.kernel_h = attr['kernel_shape'][0]
                layer.convolution_param.kernel_w = attr['kernel_shape'][1]

        kwargs['group'] = attr['group']
        layer.convolution_param.group = attr['group']

        bias_name = '{0}.bias'.format(source_node.weights_name)
        weights_name = '{0}.weight'.format(source_node.weights_name)
        weight = self.state_dict[weights_name]

        weight = weight.numpy()

        self.set_weight(source_node.name, 'weights', weight)
        kwargs['kernel_shape'] = list(weight.shape)
        layer.convolution_param.num_output = list(weight.shape)[1]

        # handle bias
        if bias_name in self.state_dict:
            bias = self.state_dict[bias_name].numpy()
            self.set_weight(source_node.name, 'bias', bias)
            kwargs['use_bias'] = True
            layer.convolution_param.bias_term = True
            layer.blobs.extend([as_blob(weight), as_blob(bias)])
        else:
            kwargs['use_bias'] = False
            layer.convolution_param.bias_term = False
            layer.blobs.extend([as_blob(weight)])

        for b in source_node.in_edges:
            layer.bottom.append(b)

        if len(source_node.in_edges) == 0:
            layer.bottom.append(self.bottoms.pop(0))

        layer.top.append(source_node.name)

        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)
        return layer     