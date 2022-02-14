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
    'onnx::AveragePool': 'AvgPool',
    'onnx::GlobalAveragePool': 'AvgPooling',
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
    'onnx::Constant': 'Constant',
    'onnx::Reshape': 'Reshape',
    'onnx::Split': 'Split',
    'onnx::LpNormalization': 'LpNormalization',
    'prim::Constant': 'Constant',
    'onnx::LeakyRelu': 'LeakyRelu',
    'onnx::Resize': 'Resize',
    'onnx::ReduceMean': 'ReduceMean',
    'onnx::BilinearInterpolate': 'BilinearInterpolate',
    'onnx::Shape': 'Skip',
    'onnx::Gather': 'Skip',
    'onnx::Sub': 'Skip',
    'onnx::MaxUnpool': 'MaxUnPool'
}

    ############
    # property #
    ############

    @property
    def src_graph(self):
        return self.pytorch_graph


    ####################
    # Public Functions #
    ####################

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
        self.named_type = dict()
        self.caffe_net = []
        self.bottoms = list()
        self.skip_layer = dict()

    def fuse_all_conv_bn(self, model):
        stack = []
        for name, module in model.named_children():
            if list(module.named_children()):
                self.fuse_all_conv_bn(module)
                
            if isinstance(module, nn.BatchNorm2d):
                if isinstance(stack[-1][1], nn.Conv2d):
                    setattr(model, stack[-1][0], fuse_conv_bn_eval(stack[-1][1], module))
                    setattr(model, name, nn.Identity())
            else:
                stack.append((name, module))

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

        for layer in self.src_graph.topological_sort:
            current_node = self.src_graph.get_node(layer)
            onnx_node_type = current_node.type
            node_type = PytorchParser.layer_map[onnx_node_type]

            if hasattr(self, "rename_" + node_type):
                func = getattr(self, "rename_" + node_type)
                layer_data = func(current_node)
                if layer_data == None:
                    continue
                elif(node_type == "BatchNormalization"):
                    self.caffe_net.append(layer_data[0])
                    self.caffe_net.append(layer_data[1])
                    self.named_type[layer_data[0].name] = layer_data[0]
                    self.named_type[layer_data[1].name] = layer_data[1]                  
                else:
                    self.caffe_net.append(layer_data)                    
                    self.named_type[layer_data.name] = layer_data
            else:
                self.rename_UNKNOWN(current_node)

        text_net = pb2.NetParameter()
        binary_weights = pb2.NetParameter()
        binary_weights.CopyFrom(text_net)

        if self.fuse:
            for layer in self.caffe_net:
                if layer.type in ["ReLU"] and self.named_type[layer.bottom[0]].type == "Convolution":
                    self.named_type[layer.bottom[0]].top[0] = layer.top[0]                
                    layer.bottom[0] = layer.top[0]

        for layer in self.caffe_net:
            binary_weights.layer.extend([layer])
            layer_proto = pb2.LayerParameter()
            layer_proto.CopyFrom(layer)
            del layer_proto.blobs[:]
            text_net.layer.extend([layer_proto])

        return text_net, binary_weights

    ##########
    # Layers #
    ##########
    def rename_Skip(self, source_node):
        print("PyTorch parser will skip operator [%s] with name [%s]."
              % (source_node.type, source_node.name)) 
   
        attr = source_node.attrs

        if 'value' in attr:
            self.skip_layer[source_node.real_name] = attr['value'].numpy()
        else:    
            self.skip_layer[source_node.real_name] = None        

        return None

    def rename_Data(self, shape, name):
        layer = pb2.LayerParameter()
        layer.type = 'Input'
        input_shape = pb2.BlobShape()
        input_shape.dim.extend(shape)
        layer.input_param.shape.extend([input_shape])
        layer.top.append(name)
        layer.name = name
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

        return layer

    def rename_AvgPooling(self, source_node):
        attr = source_node.attrs
        kwargs = dict()
        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.AVE
        layer.pooling_param.global_pooling = True
        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name

        return layer

    def rename_Sigmoid(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Sigmoid"

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name

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

        layer_bn.top.append(source_node.name)

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

        layer_scale.bottom.append(source_node.real_name)

        layer_scale.top.append(source_node.name)

        layer_scale.name = source_node.real_name + "_scale"

        return [layer_bn, layer_scale]
        # return layer_bn

    def rename_Relu(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "ReLU"

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name

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

        return layer

    def rename_Add(self, source_node):
        attr = source_node.attrs

        layer = pb2.LayerParameter()
        layer.type = "Eltwise"

        skip_all = False
        for b in source_node.in_edges:
            if b in self.skip_layer:
                skip_all = True
            else:
                skip_all = False
                layer.bottom.append(b)

        if skip_all:
            self.skip_layer[source_node.real_name] = self.skip_layer[source_node.in_edges[-1]]

            return None  

        layer.top.append(source_node.name)
        layer.name = source_node.real_name

        return layer

    def rename_AvgPool(self, source_node):
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

        return layer

    def rename_Flatten(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        layer.type = "Flatten"

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name

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

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name

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

        return layer

    def rename_Upsample(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        layer.type = "Upsample"

        for b in source_node.in_edges:
            if b in self.skip_layer.keys():
                if not isinstance(self.skip_layer[b], type(None)):
                    layer.upsample_param.scale = int(self.skip_layer[b][0])
                continue
            layer.bottom.append(b)
        layer.top.append(source_node.name)

        layer.name = source_node.real_name
        return layer

    def rename_Concat(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        layer.type = "Concat"
        layer.concat_param.axis = attr['axis']
        
        skip_all = False
        for b in source_node.in_edges:
            if b in self.skip_layer:
                skip_all = True
            else:
                skip_all = False
                layer.bottom.append(b)

        if skip_all:
            self.skip_layer[source_node.real_name] = self.skip_layer[source_node.in_edges[-1]]

            return None                

        layer.top.append(source_node.name)

        layer.name = source_node.real_name

        return layer

    def rename_Unsqueeze(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        layer.type = "Unsqueeze"

        if 'axes' in attr:
            layer.unsqueeze_param.dim = attr['axes'][0]
        else:
            layer.unsqueeze_param.dim = 0            

        skip_all = False
        for b in source_node.in_edges:
            if b in self.skip_layer:
                skip_all = True
            else:
                skip_all = False
                layer.bottom.append(b)

        if skip_all:
            self.skip_layer[source_node.real_name] = self.skip_layer[source_node.in_edges[-1]]

            return None  

        layer.top.append(source_node.name)

        layer.name = source_node.real_name
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
        return layer     

    def rename_Pad(self, source_node):
        attr = source_node.attrs

        layer = pb2.LayerParameter()
        layer.type = "Pad"
        if 'pads' in attr:
        # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
            layer.pad_param.pad_u = attr['pads'][2]
            layer.pad_param.pad_d = attr['pads'][6]
            layer.pad_param.pad_l = attr['pads'][3]
            layer.pad_param.pad_r = attr['pads'][7]

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)

        layer.name = source_node.real_name
        return layer   

    def rename_HardSwish(self, source_node):
        attr = source_node.attrs

        layer = pb2.LayerParameter()
        layer.type = "HardSwish"

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)

        layer.name = source_node.real_name
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
        return layer            

    def rename_Mul(self, source_node):
        attr = source_node.attrs

        layer = pb2.LayerParameter()
        layer.type = "Eltwise"

        layer.eltwise_param.operation = 0

        skip_all = False
        for b in source_node.in_edges:
            if b in self.skip_layer:
                skip_all = True
            else:
                skip_all = False
                layer.bottom.append(b)

        if skip_all:
            self.skip_layer[source_node.real_name] = self.skip_layer[source_node.in_edges[-1]]

            return None  

        layer.top.append(source_node.name)
        layer.name = source_node.real_name

        return layer

    def rename_Slice(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        layer.type = "Slice"

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name

        return layer        

    def rename_Constant(self, source_node):
        attr = source_node.attrs

        if 'value' in attr:
            self.skip_layer[source_node.real_name] = attr['value'].numpy()
        else:    
            self.skip_layer[source_node.real_name] = None        

        return None

    def rename_Reshape(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        layer.type = "Reshape"

        if 'shape' in attr:
            for each in attr['shape']:
                layer.reshape_param.shape.dim.extend([each])

        for b in source_node.in_edges:
            if b in self.skip_layer.keys():
                if not isinstance(self.skip_layer[b], type(None)):
                    for each in self.skip_layer[b]:
                        layer.reshape_param.shape.dim.extend([each])
                continue
            layer.bottom.append(b)

        layer.top.append(source_node.name)

        layer.name = source_node.real_name
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

        return layer             

    def rename_Resize(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        layer.type = "Upsample"

        for b in source_node.in_edges:
            if b in self.skip_layer.keys():
                if not isinstance(self.skip_layer[b], type(None)):
                    layer.upsample_param.scale = int(self.skip_layer[b][0])
                continue
            layer.bottom.append(b)

        layer.top.append(source_node.name)

        layer.name = source_node.real_name
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

        return layer

    def rename_ReduceMean(self, source_node):
        attr = source_node.attrs
        kwargs = dict()
        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.AVE
        layer.pooling_param.global_pooling = True
        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name

        return layer      

    def rename_BilinearInterpolate(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        if self.opset_version == 9:
            layer.type = "BilinearInterpolate"
            layer.bilinear_interpolate_param.align_corners = attr['align_corners']

            for b in source_node.in_edges:
                if b in self.skip_layer.keys():
                    if not isinstance(self.skip_layer[b], type(None)):
                        layer.bilinear_interpolate_param.scale_factor = int(self.skip_layer[b][0])
                    continue
                layer.bottom.append(b)

            layer.top.append(source_node.name)
            layer.name = source_node.real_name
        
            return layer  
        else:
            raise Exception('Unsupported opset_version: {}'.format(self.opset_versionc))

    def rename_MaxUnPool(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()

        layer.type = "MaxUnPool"
        layer.max_unpool_param.dst_h = source_node.output_shape[2]
        layer.max_unpool_param.dst_w = source_node.output_shape[3]

        for b in source_node.in_edges:
            if b in self.skip_layer.keys():
                continue
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name
    
        return layer  
     