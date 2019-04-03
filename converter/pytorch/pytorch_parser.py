#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import os
import numpy as np
from converter.core.parser import Parser
from converter.pytorch.pytorch_graph import PytorchGraph
import caffe.proto.caffe_pb2 as pb2
# import caffe_pb2 as pb2
import torch
import torchvision

global caffe_net

caffe_net = []

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
    'onnx::Flatten': 'Flatten',
    'onnx::Gemm': 'FullyConnected',
    'onnx::Dropout': 'Dropout',
    'onnx::LogSoftmax': 'Softmax',
    'onnx::Transpose': 'Permute',
    'onnx::Constant': 'Constant',

    'aten::max_pool2d': 'MaxPooling',
    'aten::adaptive_avg_pool2d': 'AvgPooling'

    # TODO
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

    def __init__(self, model_file_name, input_shape):
        super(PytorchParser, self).__init__()
        if not os.path.exists(model_file_name):
            print("Pytorch model file [{}] is not found.".format(model_file_name))
            assert False
        # test

        # cpu: https://github.com/pytorch/pytorch/issues/5286
        try:
            model = torch.load(model_file_name, map_location='cpu')
        except:
            model = torch.load(model_file_name, map_location='cpu')

        self.weight_loaded = True

        # Build network graph
        self.pytorch_graph = PytorchGraph(model)
        self.input_shape = tuple([1] + input_shape)
        self.pytorch_graph.build(self.input_shape)
        self.state_dict = self.pytorch_graph.state_dict
        self.shape_dict = self.pytorch_graph.shape_dict


    def gen_IR(self):

        bottoms = []
        top = []
        for layer in self.src_graph.topological_sort:
            current_node = self.src_graph.get_node(layer)
            onnx_node_type = current_node.type
            node_type = PytorchParser.layer_map[onnx_node_type]

            if len(bottoms) == 0:
                func = getattr(self, "rename_Data")
                layer_data = func()
                caffe_net.append(layer_data)
                bottoms.append('data')

            if hasattr(self, "rename_" + node_type):
                func = getattr(self, "rename_" + node_type)
                layer_data = func(current_node)
                if(node_type == "BatchNormalization"):
                    caffe_net.append(layer_data[0])
                    caffe_net.append(layer_data[1])
                    # caffe_net.append(layer_data)
                else:
                    caffe_net.append(layer_data)

            else:
                self.rename_UNKNOWN(current_node)

        text_net = pb2.NetParameter()

        binary_weights = pb2.NetParameter()
        binary_weights.CopyFrom(text_net)
        for layer in caffe_net:
            binary_weights.layer.extend([layer])
            layer_proto = pb2.LayerParameter()
            layer_proto.CopyFrom(layer)
            del layer_proto.blobs[:]
            text_net.layer.extend([layer_proto])

        return text_net, binary_weights

    ##########
    # Layers #
    ##########

    def rename_UNKNOWN(self, source_node):
        print (source_node.layer)
        print (source_node.layer.data.size())
        assert False
        print("PyTorch parser has not supported operator [%s] with name [%s]."
              % (source_node.type, source_node.name))

    def rename_Data(self):
        layer = pb2.LayerParameter()
        layer.type = 'Input'
        input_shape = pb2.BlobShape()
        input_shape.dim.extend(self.input_shape)
        layer.input_param.shape.extend([input_shape])
        layer.top.append("data")
        layer.name = "data"
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
            layer.bottom.append("data")

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

    def rename_MaxPooling(self, source_node):
        attr = source_node.attrs
        kwargs = dict()
        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.MAX

        if len(attr['padding']) == 4:
            kwargs['padding'] = [0] + attr['padding'][0:2] + [0, 0] + attr['padding'][2:] + [0]
            if attr['padding'][0] == attr['padding'][1]:
                layer.pooling_param.pad = [attr['padding'][0]]
            else:
                layer.pooling_param.pad_h = attr['padding'][0]
                layer.pooling_param.pad_w = attr['padding'][1]
        elif len(attr['padding']) == 2:
            kwargs['padding'] = ([0] + attr['padding'][0:2] + [0]) * 2
            if attr['padding'][0] == attr['padding'][1]:
                layer.pooling_param.pad = attr['padding'][0]
            else:
                layer.pooling_param.pad_h = attr['padding'][0]
                layer.pooling_param.pad_w = attr['padding'][1]

        if 'stride' not in attr:
            kwargs['stride'] = [1] + [1, 1] + [1]
            layer.pooling_param.stride = 1
        else:
            kwargs['stride'] = [1] + attr['stride'] + [1]
            if attr['stride'][0] == attr['stride'][1]:
                layer.pooling_param.stride = attr['stride'][0]
            else:
                layer.pooling_param.stride_h = attr['stride'][0]
                layer.pooling_param.stride_w = attr['stride'][1]

        if 'kernel_size' not in attr:
            kwargs['kernel_size'] = [1] + [1, 1] + [1]
            layer.pooling_param.kernel_size.extend(1)
        else:
            kwargs['kernel_size'] = [1] + attr['kernel_size'] + [1]
            if attr['kernel_size'][0] == attr['kernel_size'][1]:
                layer.pooling_param.kernel_size = attr['kernel_size'][0]
            else:
                layer.pooling_param.kernel_h = attr['kernel_size'][0]
                layer.pooling_param.kernel_w = attr['kernel_size'][1]

        if 'ceil_mode' not in attr:
            kwargs['ceil_mode'] = 0
            if attr['padding'][0] == attr['padding'][1]:
                if attr['stride'][0] > 1 and attr['padding'][0] > 0:
                    layer.pooling_param.pad = attr['padding'][0] - 1
            else:
                if attr['stride'][0] > 1 and attr['padding'][0] > 0:
                    layer.pooling_param.pad_h = attr['padding'][0] - 1
                if attr['stride'][1] > 1 and attr['padding'][1] > 0:
                    layer.pooling_param.pad_w = attr['padding'][1] - 1
        else:
            if attr['ceil_mode'] != 1:
                if attr['padding'][0] == attr['padding'][1]:
                    if attr['stride'][0] > 1 and attr['padding'][0] > 0:
                        layer.pooling_param.pad = attr['padding'][0] - 1
                else:
                    if attr['stride'][0] > 1 and attr['padding'][0] > 0:
                        layer.pooling_param.pad_h = attr['padding'][0] - 1
                    if attr['stride'][1] > 1 and attr['padding'][1] > 0:
                        layer.pooling_param.pad_w = attr['padding'][1] - 1

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
            kwargs['ceil_mode'] = 0
            if attr['pads'][0] == attr['pads'][1]:
                if attr['strides'][0] > 1 and attr['pads'][0] > 0:
                    layer.pooling_param.pad = attr['pads'][0] - 1
            else:
                if attr['strides'][0] > 1 and attr['pads'][0] > 0:
                    layer.pooling_param.pad_h = attr['pads'][0] - 1
                if attr['strides'][1] > 1 and attr['pads'][1] > 0:
                    layer.pooling_param.pad_w = attr['pads'][1] - 1
        else:
            if attr['ceil_mode'] != 1:
                if attr['padding'][0] == attr['pads'][1]:
                    if attr['strides'][0] > 1 and attr['pads'][0] > 0:
                        layer.pooling_param.pad = attr['pads'][0] - 1
                else:
                    if attr['strides'][0] > 1 and attr['pads'][0] > 0:
                        layer.pooling_param.pad_h = attr['pads'][0] - 1
                    if attr['strides'][1] > 1 and attr['pads'][1] > 0:
                        layer.pooling_param.pad_w = attr['pads'][1] - 1

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name

        return layer

    def rename_Add(self, source_node):
        attr = source_node.attrs

        layer = pb2.LayerParameter()
        layer.type = "Eltwise"

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name

        return layer

    def rename_AvgPool(self, source_node):
        attr = source_node.attrs
        kwargs = dict()
        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.AVE

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
        # train_only = pb2.NetStateRule()
        # train_only.phase = pb2.TEST
        # layer.exclude.extend([train_only])

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

        weights_name = '{0}.weight'.format(source_node.weights_name)

        weight = self.state_dict[weights_name]

        weight = weight.numpy()

        layer.blobs.extend([as_blob(weight[0])])

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)

        layer.name = source_node.real_name

        return layer

    def rename_Constant(self, source_node):
        kwargs = dict()
        layer = pb2.LayerParameter()
        layer.type = "Normalize"

        weights_name = '{0}.weight'.format(source_node.weights_name)

        weight = self.state_dict[weights_name]

        weight = weight.numpy()

        layer.blobs.extend([as_blob(weight[0])])

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)

        layer.name = source_node.real_name

        return layer