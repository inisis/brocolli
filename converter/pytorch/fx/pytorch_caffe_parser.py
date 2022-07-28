#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
import re
import numpy as np
from loguru import logger
from converter.pytorch.fx.pytorch_graph import PytorchGraph
import caffe.proto.caffe_pb2 as pb2

import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval
from torch.fx.node import Node
from torch.fx.graph_module import GraphModule
import google.protobuf.text_format
from .utils import get_function_name


def as_blob(array):
    blob = pb2.BlobProto()
    blob.shape.dim.extend(array.shape)
    blob.data.extend(array.astype(float).flat)
    return blob

class PytorchCaffeParser():
    def __init__(self, model, input_shape, fuse=False, concrete_args=None):
        super(PytorchCaffeParser, self).__init__()
        self.fuse = fuse
        self.model = model
        self.input_shape = input_shape
        self.concrete_args = concrete_args

        if isinstance(self.model, GraphModule):
            pass
        elif isinstance(self.model, nn.Module):
            if self.fuse:
                self.fuse_all_conv_bn(self.model)
        else:
            raise Exception("model must be a torch.nn.Module or a torch.fx.GraphModule")

        self.pytorch_graph = PytorchGraph(self.model, self.input_shape, self.concrete_args)
        self.state_dict = self.pytorch_graph.trace.state_dict()
        self.modules = dict(self.pytorch_graph.trace.named_modules())
        self.layers = []

    def run(self):
        self.text_net, self.binary_weights = self.gen_ir()

    def save(self, dest_path):
        self.save_to_proto(self.text_net, dest_path + ".prototxt")
        self.save_weights(self.binary_weights, dest_path + ".caffemodel")
        logger.info("prototxt saved to {}.prototxt".format(dest_path))
        logger.info("caffemodel saved to {}.caffemodel".format(dest_path))

    def save_to_proto(self, net, filename):
        with open(filename, 'wb') as f:
            f.write(google.protobuf.text_format.MessageToString(net).encode())

    def save_weights(self, weights, filename):
        with open(filename, 'wb') as f:
            f.write(weights.SerializeToString())

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

    def list_try_get(self, list, idx, default=None):
        try:
            return list[idx]
        except IndexError:
            return default

    def recursive_find_name(self, node):
        if node.op == 'placeholder':
            return node.name
        elif node.op == 'call_module':
            module = self.modules[node.target]
            if isinstance(module, nn.Identity):
                node_ = node.args[0]
                return self.recursive_find_name(node_)              
            else:
                return node.name
        elif node.op == 'call_function':
            function_name = get_function_name(node.target)
            if function_name == "getitem":
                node_name = node.args[0].name + '_' + str(node.args[1])
                return node_name
            else:
                return node.name
        elif node.op == 'call_method':
            if str(node.target) == 'contiguous':
                node_ = node.args[0]
                return self.recursive_find_name(node_)
            else:  
                return node.name

    def add_bottom_top(self, layer, source_node):
        for node in source_node.args:
            if isinstance(node, Node):
                bottom_name = self.recursive_find_name(node)
                if bottom_name is None:
                    continue                
                layer.bottom.append(bottom_name)
            elif isinstance(node, list) or isinstance(node, tuple): # cat function args[0]
                for node_ in node:
                    if isinstance(node_, Node):
                        bottom_name = self.recursive_find_name(node_)
                        if bottom_name is None:
                            continue                        
                        layer.bottom.append(bottom_name)
            else:
                continue
        layer.top.append(source_node.name)
        layer.name = source_node.name

    def gen_ir(self):
        for node in self.pytorch_graph.nodes:
            if node.op == 'placeholder':
                func = getattr(self, "rename_Data")
                layer_data = func(node)
                self.layers.append(layer_data)
            elif node.op == 'call_module':
                module = self.modules[node.target]
                if isinstance(module, nn.Conv2d):
                    func = getattr(self, "rename_Conv")
                    layer_data = func(node, module)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.BatchNorm2d):
                    func = getattr(self, "rename_BatchNormalization")
                    layer_data = func(node, module)
                    self.layers.append(layer_data[0])
                    self.layers.append(layer_data[1])
                elif isinstance(module, nn.ReLU):
                    func = getattr(self, "rename_ReLU")
                    layer_data = func(node)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.MaxPool2d):
                    func = getattr(self, "rename_MaxPool2d")
                    layer_data = func(node, module)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.AdaptiveAvgPool2d):
                    if isinstance(module.output_size, int):
                        output_size = [1]
                        output_size_len = 1
                    else:
                        output_size = [int(v) for v in module.output_size]
                        output_size_len = len(module.output_size)
                    if output_size == [1] * output_size_len:
                        func = getattr(self, "rename_AdaptiveAvgPool2d")
                        layer_data = func(node)
                    else:
                        func = getattr(self, "rename_AveragePool")
                        layer_data = func(node, module)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.Linear):
                    func = getattr(self, "rename_Linear")
                    layer_data = func(node, module)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.Dropout):
                    func = getattr(self, "rename_Dropout")
                    layer_data = func(node, module)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.ReLU6):
                    func = getattr(self, "rename_ReLU6")
                    layer_data = func(node)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.Hardswish):
                    func = getattr(self, "rename_Hardswish")
                    layer_data = func(node)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.Hardsigmoid):
                    func = getattr(self, "rename_Hardsigmoid")
                    layer_data = func(node)
                    self.layers.append(layer_data)                    
                elif isinstance(module, nn.Identity):
                    pass
                elif isinstance(module, nn.AvgPool2d):                
                    func = getattr(self, "rename_AveragePool")
                    layer_data = func(node, module)
                    self.layers.append(layer_data)  
                elif isinstance(module, nn.SiLU):                
                    func = getattr(self, "rename_SiLU")
                    layer_data = func(node, module)
                    self.layers.append(layer_data[0]) 
                    self.layers.append(layer_data[1])                    
                elif isinstance(module, nn.Upsample):                
                    func = getattr(self, "rename_Upsample")
                    layer_data = func(node, module)
                    self.layers.append(layer_data)     
                elif isinstance(module, nn.LeakyReLU): 
                    func = getattr(self, "rename_LeakyRelu")
                    layer_data = func(node, module)
                    self.layers.append(layer_data)                                                                           
                else:
                     raise NotImplementedError("module %s is not implemented" % (module))
            elif node.op == 'call_function':
                function_name = get_function_name(node.target)
                if function_name == "relu":
                    func = getattr(self, "rename_relu")
                    layer_data = func(node)
                    self.layers.append(layer_data)
                elif function_name == "add":
                    func = getattr(self, "rename_add")
                    layer_data = func(node)
                    self.layers.append(layer_data)
                elif function_name == "flatten":
                    func = getattr(self, "rename_flatten")
                    layer_data = func(node)
                    self.layers.append(layer_data)                    
                elif function_name == "cat":
                    func = getattr(self, "rename_cat")
                    layer_data = func(node)
                    self.layers.append(layer_data)
                elif function_name == "adaptive_avg_pool2d":
                    if isinstance(node.args[1], int):
                        output_size = [1]
                        output_size_len = 1
                    else:
                        output_size = [int(v) for v in node.args[1]]
                        output_size_len = len(node.args[1])
                    if output_size == [1] * output_size_len:                    
                        func = getattr(self, "rename_adaptive_avg_pool2d")
                        layer_data = func(node)
                    else:
                        func = getattr(self, "rename_avg_pool2d")
                        layer_data = func(node)                        
                    self.layers.append(layer_data)
                elif function_name == "hardsigmoid":
                    func = getattr(self, "rename_hardsigmoid")
                    layer_data = func(node)
                    self.layers.append(layer_data)
                elif function_name == "mul":
                    func = getattr(self, "rename_mul")
                    layer_data = func(node)
                    if isinstance(layer_data, tuple):
                        for layer in layer_data:
                            self.layers.append(layer)
                    else:
                        self.layers.append(layer_data)
                elif function_name == "getitem":
                    pass
                elif function_name == "floordiv":
                    pass
                elif function_name == "transpose":
                    func = getattr(self, "rename_transpose")
                    layer_data = func(node)
                    self.layers.append(layer_data)
                elif function_name == "prelu":
                    func = getattr(self, "rename_prelu")
                    layer_data = func(node)
                    self.layers.append(layer_data)
                elif function_name == "hardtanh":
                    func = getattr(self, "rename_hardtanh")
                    layer_data = func(node)
                    self.layers.append(layer_data)      
                elif function_name == "leaky_relu":
                    func = getattr(self, "rename_leaky_relu")
                    layer_data = func(node)
                    self.layers.append(layer_data)
                elif function_name == "sigmoid":
                    func = getattr(self, "rename_sigmoid")
                    layer_data = func(node)
                    self.layers.append(layer_data)      
                elif function_name == "softmax":
                    func = getattr(self, "rename_softmax")
                    layer_data = func(node)
                    self.layers.append(layer_data)          
                elif function_name == "hardswish":
                    func = getattr(self, "rename_hardswish")
                    layer_data = func(node)
                    self.layers.append(layer_data)       
                elif function_name == "conv2d":
                    func = getattr(self, "rename_conv2d")
                    layer_data = func(node)
                    self.layers.append(layer_data)           
                elif function_name == "linear":
                    func = getattr(self, "rename_linear")
                    layer_data = func(node)
                    self.layers.append(layer_data)      
                elif function_name == "avg_pool2d":
                    func = getattr(self, "rename_avg_pool2d")
                    layer_data = func(node)
                    self.layers.append(layer_data)        
                elif function_name == "max_pool2d_with_indices":
                    func = getattr(self, "rename_max_pool2d_with_indices")
                    layer_data = func(node)
                    self.layers.append(layer_data)     
                elif function_name == "chunk":
                    func = getattr(self, "rename_chunk")
                    layer_data = func(node)
                    self.layers.append(layer_data)  
                elif function_name == "split":
                    func = getattr(self, "rename_split")
                    layer_data = func(node)
                    self.layers.append(layer_data)   
                elif function_name == "getattr":
                    pass                                                                                                                                                                                                                   
                else:
                     raise NotImplementedError("function %s is not implemented" % (function_name))
            elif node.op == 'call_method':
                if str(node.target) == 'size':
                    pass
                elif str(node.target) == 'view':
                    func = getattr(self, "rename_view")
                    layer_data = func(node)
                    self.layers.append(layer_data)
                elif str(node.target) == 'contiguous':
                    pass
                elif str(node.target) == 'chunk':
                    func = getattr(self, "rename_chunk")
                    layer_data = func(node)
                    self.layers.append(layer_data)
                elif str(node.target) == 'mean':
                    func = getattr(self, "rename_adaptive_avg_pool2d")
                    layer_data = func(node)
                    self.layers.append(layer_data)   
                elif str(node.target) == 'permute':
                    func = getattr(self, "rename_permute")
                    layer_data = func(node)
                    self.layers.append(layer_data)     
                elif str(node.target) == 'flatten':
                    func = getattr(self, "rename_view")
                    layer_data = func(node)
                    self.layers.append(layer_data) 
                elif str(node.target) == "sigmoid":
                    func = getattr(self, "rename_sigmoid")
                    layer_data = func(node)
                    self.layers.append(layer_data)     
                elif str(node.target) == "squeeze":
                    func = getattr(self, "rename_view")
                    layer_data = func(node)
                    self.layers.append(layer_data)                                                                                                               
                else:
                    raise NotImplementedError("method %s is not implemented" % (str(node.target)))
            elif node.op == 'output':
                pass
            elif node.op == 'get_attr':
                pass
            else:
                raise NotImplementedError("op type %s is not implemented" % (node.op))

        text_net = pb2.NetParameter()
        binary_weights = pb2.NetParameter()
        binary_weights.CopyFrom(text_net)

        for layer in self.layers:
            layer.name = layer.name.replace(".", "")
            binary_weights.layer.extend([layer])
            layer_proto = pb2.LayerParameter()
            layer_proto.CopyFrom(layer)
            del layer_proto.blobs[:]
            text_net.layer.extend([layer_proto])

        return text_net, binary_weights

    ##########
    # Layers #
    ##########
    def rename_Data(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = 'Input'
        input_shape = pb2.BlobShape()
        input_shape.dim.extend(source_node.meta['tensor_meta'].shape)
        layer.input_param.shape.extend([input_shape])
        layer.top.append(source_node.name)
        layer.name = source_node.name
        return layer

    def rename_Conv(self, source_node, module):
        layer = pb2.LayerParameter()
        layer.type = "Convolution"

        layer.convolution_param.dilation.extend([module.dilation[0]])

        if isinstance(module.padding, tuple):
            if module.padding[0] == module.padding[1]:
                layer.convolution_param.pad.extend([module.padding[0]])
            else:
                layer.convolution_param.pad_h = module.padding[0]
                layer.convolution_param.pad_w = module.padding[1]
        else:
            layer.convolution_param.pad.extend([module.padding])              

        if isinstance(module.stride, tuple):
            if module.stride[0] == module.stride[1]:
                layer.convolution_param.stride.extend([module.stride[0]])
            else:
                layer.convolution_param.stride_h = module.stride[0]
                layer.convolution_param.stride_w = module.stride[1]
        else:
            layer.convolution_param.stride.extend([module.stride])   

        if isinstance(module.kernel_size , tuple):
            if module.kernel_size [0] == module.kernel_size [1]:
                layer.convolution_param.kernel_size.extend([module.kernel_size[0]])
            else:
                layer.convolution_param.kernel_h = module.kernel_size [0]
                layer.convolution_param.kernel_w = module.kernel_size [1]
        else:
            layer.convolution_param.kernel_size.extend([module.kernel_size ])

        layer.convolution_param.group = module.groups 

        bias_name = '{0}.bias'.format(source_node.target)
        weights_name = '{0}.weight'.format(source_node.target)

        weight = self.state_dict[weights_name].numpy()

        layer.convolution_param.num_output = module.out_channels

        if module.bias is not None:
            bias = self.state_dict[bias_name].numpy()
            layer.convolution_param.bias_term = True
            layer.blobs.extend([as_blob(weight), as_blob(bias)])
        else:
            layer.convolution_param.bias_term = False
            layer.blobs.extend([as_blob(weight)])
        
        self.add_bottom_top(layer, source_node)

        return layer

    def rename_AdaptiveAvgPool2d(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.AVE
        layer.pooling_param.global_pooling = True
        
        self.add_bottom_top(layer, source_node)

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

    def rename_BatchNormalization(self, source_node, module):
        layer_bn = pb2.LayerParameter()
        layer_bn.type = "BatchNorm"

        layer_bn.batch_norm_param.use_global_stats = 1
        layer_bn.batch_norm_param.eps = module.eps 

        mean_name = '{0}.running_mean'.format(source_node.target)
        var_name = '{0}.running_var'.format(source_node.target)

        mean = self.state_dict[mean_name].numpy()
        variance = self.state_dict[var_name].numpy()

        layer_bn.blobs.extend([as_blob(mean), as_blob(variance), as_blob(np.array([1.]))])

        layer_bn.bottom.append(source_node.args[0].name)

        layer_bn.top.append(source_node.name + '_bn')
        layer_bn.name = source_node.name + '_bn'

        layer_scale = pb2.LayerParameter()
        layer_scale.type = "Scale"

        bias_name = '{0}.bias'.format(source_node.target)
        weights_name = '{0}.weight'.format(source_node.target)

        weight = self.state_dict[weights_name].numpy()

        if bias_name in self.state_dict:
            bias = self.state_dict[bias_name].numpy()
            layer_scale.scale_param.bias_term = True
            layer_scale.blobs.extend([as_blob(weight), as_blob(bias)])
        else:
            layer_scale.scale_param.bias_term = False
            layer_scale.blobs.extend([as_blob(weight)])

        layer_scale.bottom.append(source_node.name + '_bn')
        layer_scale.top.append(source_node.name)
        layer_scale.name = source_node.name

        return layer_bn, layer_scale

    def rename_ReLU(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "ReLU"

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_MaxPool2d(self, source_node, module):
        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.MAX
        
        if isinstance(module.padding, tuple):
            if module.padding[0] == module.padding[1]:
                layer.pooling_param.pad = module.padding[0]
            else:
                layer.pooling_param.pad_h = module.padding[0]
                layer.pooling_param.pad_w = module.padding[1]
        else:
            layer.pooling_param.pad = module.padding             

        if isinstance(module.stride, tuple):
            if module.stride[0] == module.stride[1]:
                layer.pooling_param.stride = module.stride[0]
            else:
                layer.pooling_param.stride_h = module.stride[0]
                layer.pooling_param.stride_w = module.stride[1]
        else:
            layer.pooling_param.stride = module.stride

        if isinstance(module.kernel_size, tuple):
            if module.kernel_size[0] == module.kernel_size[1]:
                layer.pooling_param.kernel_size = module.kernel_size[0]
            else:
                layer.pooling_param.kernel_h = module.kernel_size[0]
                layer.pooling_param.kernel_w = module.kernel_size[1]
        else:
            layer.pooling_param.kernel_size = module.kernel_size

        layer.pooling_param.ceil_mode = module.ceil_mode

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_Add(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Eltwise"

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_AveragePool(self, source_node, module):
        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.AVE
        if isinstance(module, nn.AdaptiveAvgPool2d):
            dim = source_node.meta['tensor_meta'].shape[2:]
            if isinstance(module.output_size, int):
                output_size = [module.output_size] * len(dim)
            else:
                output_size = module.output_size
            k = [int(dim[i] / output_size[i]) for i in range(0, len(dim))]
            if len(k) == 1:
                layer.pooling_param.stride = k[0]
            else:
                if k[0] == k[1]:
                    layer.pooling_param.stride = k[0]
                else:
                    layer.pooling_param.stride_h = k[0]
                    layer.pooling_param.stride_w = k[1]

            if len(k) == 1:
                layer.pooling_param.kernel_size.extend(k[0])
            else:
                if k[0] == k[1]:
                    layer.pooling_param.kernel_size = k[0]
                else:
                    layer.pooling_param.kernel_h = k[0]
                    layer.pooling_param.kernel_w = k[1]
            
            self.add_bottom_top(layer, source_node)

            return layer
        else:
            if isinstance(module.padding, tuple):
                if module.padding[0] == module.padding[1]:
                    layer.pooling_param.pad.extend([module.padding[0]])
                else:
                    layer.pooling_param.pad_h = module.padding[0]
                    layer.pooling_param.pad_w = module.padding[1]
            else:
                layer.pooling_param.pad = module.padding

            if isinstance(module.stride, tuple):
                if module.stride[0] == module.stride[1]:
                    layer.pooling_param.stride.extend([module.stride[0]])
                else:
                    layer.pooling_param.stride_h = module.stride[0]
                    layer.pooling_param.stride_w = module.stride[1]
            else:
                layer.pooling_param.stride = module.stride

            if isinstance(module.kernel_size, tuple):
                if module.kernel_size[0] == module.kernel_size[1]:
                    layer.pooling_param.kernel_size.extend([module.kernel_size[0]])
                else:
                    layer.pooling_param.kernel_h = module.kernel_size[0]
                    layer.pooling_param.kernel_w = module.kernel_size[1]
            else:
                layer.pooling_param.kernel_size = module.kernel_size

            layer.pooling_param.ceil_mode = module.ceil_mode 

            self.add_bottom_top(layer, source_node)

            return layer

    def rename_Flatten(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Flatten"

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_Linear(self, source_node, module):
        layer = pb2.LayerParameter()
        layer.type = "InnerProduct"

        bias_name = '{0}.bias'.format(source_node.target)
        weights_name = '{0}.weight'.format(source_node.target)

        weight = self.state_dict[weights_name].numpy()

        if module.bias is not None:
            bias = self.state_dict[bias_name].numpy()
            layer.inner_product_param.bias_term = True
            layer.blobs.extend([as_blob(weight), as_blob(bias)])
        else:
            layer.inner_product_param.bias_term = False
            layer.blobs.extend([as_blob(weight)])

        layer.inner_product_param.num_output = module.out_features 

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_Dropout(self, source_node, module):
        layer = pb2.LayerParameter()
        layer.type = "Dropout"

        layer.dropout_param.dropout_ratio = module.p

        self.add_bottom_top(layer, source_node)

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
        layer = pb2.LayerParameter()
        layer.type = "Permute"

        for arg in source_node.args:
            if isinstance(arg, int):
                layer.permute_param.order.extend([arg])

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_Upsample(self, source_node, module):
        layer = pb2.LayerParameter()
        layer.type = "Upsample"

        layer.upsample_param.scale = int(module.scale_factor)

        self.add_bottom_top(layer, source_node)
   
        return layer

    def rename_Cat(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Concat"

        if 'dim' in source_node.kwargs:
            layer.concat_param.axis = source_node.kwargs['dim']
        else:
            layer.concat_param.axis = source_node.args[1]

        self.add_bottom_top(layer, source_node)

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

    def rename_ReLU6(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "ReLU6"

        layer.relu6_param.threshold = 6

        self.add_bottom_top(layer, source_node)      

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

        layer.bottom.append(source_node.in_edges[0])
        
        layer.top.append(source_node.name)

        layer.name = source_node.real_name

        if self.is_main(layer.bottom):
            self.main_layers.append(layer)

        return layer   

    def rename_Hardswish(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "HardSwish"

        self.add_bottom_top(layer, source_node)

        return layer      

    def rename_Hardsigmoid(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "HardSigmoid"
        layer.hardsigmoid_param.alpha = 1.0 / 6
        layer.hardsigmoid_param.beta = 0.5

        self.add_bottom_top(layer, source_node)

        return layer          

    def rename_Mul(self, source_node):
        def add_flatten_before_mul(node_name, first_input, second_input):
            # first input (N,C,H,W); first input (N,C)
            layer_flatten = pb2.LayerParameter()
            layer_flatten.type = "Flatten"
            layer_flatten.flatten_param.axis = 1
            layer_flatten.bottom.append(second_input.name)
            layer_flatten.top.append(node_name + '_flatten')
            layer_flatten.name = node_name + '_flatten'

            layer_scale = pb2.LayerParameter()
            layer_scale.type = "Scale"
            layer_scale.scale_param.axis = 0
            layer_scale.bottom.append(first_input.name)            
            layer_scale.bottom.append(node_name + '_flatten')
            layer_scale.top.append(node_name)
            layer_scale.name = node_name

            return layer_flatten, layer_scale

        if list(source_node.args[0].meta['tensor_meta'].shape)[-2:] == [1, 1]:
            return add_flatten_before_mul(source_node.name, source_node.args[1], source_node.args[0])
        elif list(source_node.args[1].meta['tensor_meta'].shape)[-2:] == [1, 1]:
            return add_flatten_before_mul(source_node.name, source_node.args[0], source_node.args[1])
        else:
            layer = pb2.LayerParameter()
            layer.type = "Scale"
            layer.scale_param.axis = 0
            self.add_bottom_top(layer, source_node)

            return layer                

    def rename_Slice(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Slice"

        layer.bottom.append(source_node.in_edges[0])

        layer.top.append(source_node.name)
        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)
        return layer        

    def rename_view(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Reshape"

        for shape in source_node.meta['tensor_meta'].shape:
            layer.reshape_param.shape.dim.extend([shape])

        bottom_name = self.recursive_find_name(source_node.args[0])
        layer.bottom.append(bottom_name)
        layer.top.append(source_node.name)
        layer.name = source_node.name   

        return layer        

    def rename_Split(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Slice"

        if 'dim' in source_node.kwargs:
            layer.slice_param.axis = source_node.kwargs['dim']
        else:
            layer.slice_param.axis = source_node.args[1]

        sum_ = 0
        for idx in range(len(source_node.meta['tensor_meta'])-1):
            tensor_meta = source_node.meta['tensor_meta'][idx]
            sum_ = sum_ + tensor_meta.shape[layer.slice_param.axis]
            layer.slice_param.slice_point.extend([sum_])

        bottom_name = self.recursive_find_name(source_node.args[0])
        layer.bottom.append(bottom_name)
        for idx in range(len(source_node.meta['tensor_meta'])):
            layer.top.append(source_node.name+'_'+str(idx))
        layer.name = source_node.name

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

        if attr['mode'] == "nearest":
            layer.type = "Upsample"
            if 'scale_factor' in attr:
                layer.upsample_param.scale = attr['scale_factor'][0]

            layer.bottom.append(source_node.in_edges[0])

            layer.top.append(source_node.name)
            layer.name = source_node.real_name
            if self.is_main(layer.bottom):
                self.main_layers.append(layer)        
            return layer

        elif attr['mode'] == "linear":
            layer.type = "BilinearInterpolate"

            if attr['coordinate_transformation_mode'] == "pytorch_half_pixel":
                layer.bilinear_interpolate_param.align_corners = False
            elif attr['coordinate_transformation_mode'] == "align_corners":
                layer.bilinear_interpolate_param.align_corners = True
            else:
                raise Exception('Unsupported mode: {}'.format(attr['coordinate_transformation_mode']))

            if 'scale_factor' in attr:
                layer.bilinear_interpolate_param.scale_factor = attr['scale_factor'][0]
            else:
                layer.bilinear_interpolate_param.dst_h = attr['size'][0]
                layer.bilinear_interpolate_param.dst_w = attr['size'][1]

            layer.bottom.append(source_node.in_edges[0])

            layer.top.append(source_node.name)
            layer.name = source_node.real_name
            if self.is_main(layer.bottom):
                self.main_layers.append(layer)
            return layer
        else:         
            raise Exception('Unsupported mode: {}'.format(attr['mode']))

    def rename_LeakyRelu(self, source_node, module):
        layer = pb2.LayerParameter()
        layer.type = "ReLU"

        layer.relu_param.negative_slope = module.negative_slope

        self.add_bottom_top(layer, source_node)

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

            if 'scale' in attr:
                layer.bilinear_interpolate_param.scale_factor = attr['scale'][0]
            else:
                layer.bilinear_interpolate_param.dst_h = attr['size'][0]
                layer.bilinear_interpolate_param.dst_w = attr['size'][1]

            layer.bottom.append(source_node.in_edges[0])

            layer.top.append(source_node.name)
            layer.name = source_node.real_name
            if self.is_main(layer.bottom):
                self.main_layers.append(layer)
            return layer  
        else:
            raise Exception('Unsupported opset_version: {}'.format(self.opset_version))

    def rename_MaxUnPool(self, source_node):
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
        layer = pb2.LayerParameter()
        layer.type = "Deconvolution"

        # dilation
        if 'dilations' in attr:
            layer.convolution_param.dilation.extend([attr['dilations'][0]])
        else:
            layer.convolution_param.dilation.extend(1)

        if len(attr['pads']) == 4:
            if attr['pads'][0] == attr['pads'][1]:
                layer.convolution_param.pad.extend([attr['pads'][0]])
            else:
                layer.convolution_param.pad_h = attr['pads'][0]
                layer.convolution_param.pad_w = attr['pads'][1]
        elif len(attr['pads']) == 2:
            if attr['pads'][0] == attr['pads'][1]:
                layer.convolution_param.pad.extend([attr['pads'][0]])
            else:
                layer.convolution_param.pad_h = attr['pads'][0]
                layer.convolution_param.pad_w = attr['pads'][1]

        if 'strides' not in attr:
            layer.convolution_param.stride.extend([1])
        else:
            if attr['strides'][0] == attr['strides'][1]:
                layer.convolution_param.stride.extend([attr['strides'][0]])
            else:
                layer.convolution_param.stride_h = attr['strides'][0]
                layer.convolution_param.stride_w = attr['strides'][1]

        if 'kernel_shape' not in attr:
            layer.convolution_param.kernel_size.extend([1])
        else:
            if attr['kernel_shape'][0] == attr['kernel_shape'][1]:
                layer.convolution_param.kernel_size.extend([attr['kernel_shape'][0]])
            else:
                layer.convolution_param.kernel_h = attr['kernel_shape'][0]
                layer.convolution_param.kernel_w = attr['kernel_shape'][1]

        layer.convolution_param.group = attr['group']

        if source_node.weights_name == "":
            bias_name = 'bias'
            weights_name = 'weight'
        else:
            bias_name = '{0}.bias'.format(source_node.weights_name)
            weights_name = '{0}.weight'.format(source_node.weights_name)

        weight = self.state_dict[weights_name]

        weight = weight.numpy()

        self.set_weight(source_node.name, 'weights', weight)
        layer.convolution_param.num_output = list(weight.shape)[1]

        # handle bias
        if bias_name in self.state_dict:
            bias = self.state_dict[bias_name].numpy()
            self.set_weight(source_node.name, 'bias', bias)
            layer.convolution_param.bias_term = True
            layer.blobs.extend([as_blob(weight), as_blob(bias)])
        else:
            layer.convolution_param.bias_term = False
            layer.blobs.extend([as_blob(weight)])

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)

        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)
        return layer

    def rename_PixelShuffle(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        layer.type = "PixelShuffle"

        layer.pixelshuffle_param.upscale_factor = int(attr['blocksize'])

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name
        if self.is_main(layer.bottom):
            self.main_layers.append(layer)
        return layer

    def rename_SiLU(self, source_node, module):
        layer_sigmoid = pb2.LayerParameter()
        layer_sigmoid.type = "Sigmoid"
        layer_sigmoid.bottom.append(self.recursive_find_name(source_node.args[0]))

        layer_sigmoid.top.append(source_node.name + '_sigmoid')
        layer_sigmoid.name = source_node.name + '_sigmoid'

        layer_scale = pb2.LayerParameter()
        layer_scale.type = "Scale"
        layer_scale.scale_param.axis = 0

        layer_scale.bottom.append(self.recursive_find_name(source_node.args[0]))
        layer_scale.bottom.append(source_node.name + '_sigmoid')
        layer_scale.top.append(source_node.name)
        layer_scale.name = source_node.name

        return layer_sigmoid, layer_scale

    def rename_hardtanh(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "ReLU6"

        layer.relu6_param.threshold = source_node.kwargs['max_val']

        self.add_bottom_top(layer, source_node)   

        return layer      

    def rename_flatten(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Flatten"

        self.add_bottom_top(layer, source_node)

        return layer           

    def rename_relu(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "ReLU"

        self.add_bottom_top(layer, source_node)

        return layer      

    def rename_add(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Eltwise"

        self.add_bottom_top(layer, source_node)

        return layer   

    def rename_cat(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Concat"

        if 'dim' in source_node.kwargs:
            layer.concat_param.axis = source_node.kwargs['dim']
        else:
            layer.concat_param.axis = source_node.args[1]

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_adaptive_avg_pool2d(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.AVE
        layer.pooling_param.global_pooling = True
        
        self.add_bottom_top(layer, source_node)

        return layer   

    def rename_hardsigmoid(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "HardSigmoid"
        layer.hardsigmoid_param.alpha = 1.0 / 6
        layer.hardsigmoid_param.beta = 0.5

        self.add_bottom_top(layer, source_node)

        return layer                 

    def rename_mul(self, source_node):
        def add_flatten_before_mul(node_name, first_input, second_input):
            # first input (N,C,H,W); first input (N,C)
            layer_flatten = pb2.LayerParameter()
            layer_flatten.type = "Flatten"
            layer_flatten.flatten_param.axis = 1
            layer_flatten.bottom.append(second_input.name)
            layer_flatten.top.append(node_name + '_flatten')
            layer_flatten.name = node_name + '_flatten'

            layer_scale = pb2.LayerParameter()
            layer_scale.type = "Scale"
            layer_scale.scale_param.axis = 0
            layer_scale.bottom.append(first_input.name)            
            layer_scale.bottom.append(node_name + '_flatten')
            layer_scale.top.append(node_name)
            layer_scale.name = node_name

            return layer_flatten, layer_scale

        if list(source_node.args[0].meta['tensor_meta'].shape)[-2:] == [1, 1]:
            return add_flatten_before_mul(source_node.name, source_node.args[1], source_node.args[0])
        elif list(source_node.args[1].meta['tensor_meta'].shape)[-2:] == [1, 1]:
            return add_flatten_before_mul(source_node.name, source_node.args[0], source_node.args[1])
        else:
            layer = pb2.LayerParameter()
            layer.type = "Scale"
            layer.scale_param.axis = 0
            self.add_bottom_top(layer, source_node)

            return layer           

    def rename_permute(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Permute"

        for arg in source_node.args:
            if isinstance(arg, int):
                layer.permute_param.order.extend([arg])

        self.add_bottom_top(layer, source_node)

        return layer                        

    def rename_leaky_relu(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "ReLU"

        layer.relu_param.negative_slope = source_node.kwargs['negative_slope']

        self.add_bottom_top(layer, source_node)

        return layer        

    def rename_sigmoid(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Sigmoid"

        self.add_bottom_top(layer, source_node)

        return layer        

    def rename_softmax(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = 'Softmax'

        dim = source_node.kwargs['dim']
        if dim is None:
            stacklevel = 3
            dim = nn.functional._get_softmax_dim("softmax", len(source_node.args[0].meta['tensor_meta'].shape), stacklevel)

        layer.softmax_param.axis = dim

        self.add_bottom_top(layer, source_node)

        return layer        

    def rename_hardswish(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "HardSwish"

        self.add_bottom_top(layer, source_node)

        return layer     

    def rename_conv2d(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Convolution"

        weight = self.model.weight.detach().numpy()
        bias = self.model.bias
        stride  = source_node.args[3]
        padding = source_node.args[4]
        dilation  = source_node.args[5]
        groups  = source_node.args[6]

        layer.convolution_param.dilation.extend([dilation[0]])

        if isinstance(padding, tuple):
            if padding[0] == padding[1]:
                layer.convolution_param.pad.extend([padding[0]])
            else:
                layer.convolution_param.pad_h = padding[0]
                layer.convolution_param.pad_w = padding[1]
        else:
            layer.convolution_param.pad.extend([padding])              

        if isinstance(stride, tuple):
            if stride[0] == stride[1]:
                layer.convolution_param.stride.extend([stride[0]])
            else:
                layer.convolution_param.stride_h = stride[0]
                layer.convolution_param.stride_w = stride[1]
        else:
            layer.convolution_param.stride.extend([stride])   

 
        if weight.shape[2] == weight.shape[3]:
            layer.convolution_param.kernel_size.extend([weight.shape[2]])
        else:
            layer.convolution_param.kernel_h = weight.shape[2]
            layer.convolution_param.kernel_w = weight.shape[3]

        layer.convolution_param.group = groups 

        layer.convolution_param.num_output = weight.shape[0]

        if bias is not None:
            bias = bias.detach().numpy()
            layer.convolution_param.bias_term = True
            layer.blobs.extend([as_blob(weight), as_blob(bias)])
        else:
            layer.convolution_param.bias_term = False
            layer.blobs.extend([as_blob(weight)])
        
        bottom_name = self.recursive_find_name(source_node.args[0])
        layer.bottom.append(bottom_name)
        layer.top.append(source_node.name)
        layer.name = source_node.name   

        return layer              

    def rename_linear(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "InnerProduct"

        weight = self.model.weight.detach().numpy()
        bias = self.model.bias

        if bias is not None:
            bias = bias.detach().numpy()
            layer.inner_product_param.bias_term = True
            layer.blobs.extend([as_blob(weight), as_blob(bias)])
        else:
            layer.inner_product_param.bias_term = False
            layer.blobs.extend([as_blob(weight)])

        layer.inner_product_param.num_output = weight.shape[0]

        bottom_name = self.recursive_find_name(source_node.args[0])
        layer.bottom.append(bottom_name)
        layer.top.append(source_node.name)
        layer.name = source_node.name   

        return layer        

    def rename_avg_pool2d(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.AVE
        function_name = get_function_name(source_node.target)
        if function_name == 'adaptive_avg_pool2d':
            output_size = source_node.args[1]
            dim = source_node.args[0].meta['tensor_meta'].shape[2:] # get input shape
            if isinstance(output_size, int):
                output_size = [output_size] * len(dim)
            else:
                output_size = output_size
            k = [int(dim[i] / output_size[i]) for i in range(0, len(dim))]
            if len(k) == 1:
                layer.pooling_param.stride = k[0]
            else:
                if k[0] == k[1]:
                    layer.pooling_param.stride = k[0]
                else:
                    layer.pooling_param.stride_h = k[0]
                    layer.pooling_param.stride_w = k[1]

            if len(k) == 1:
                layer.pooling_param.kernel_size.extend(k[0])
            else:
                if k[0] == k[1]:
                    layer.pooling_param.kernel_size = k[0]
                else:
                    layer.pooling_param.kernel_h = k[0]
                    layer.pooling_param.kernel_w = k[1]
            
            self.add_bottom_top(layer, source_node)

            return layer
        else:
            kernel_size = source_node.args[1]
            stride  = self.list_try_get(source_node.args, 2, kernel_size)
            padding  = self.list_try_get(source_node.args, 3, 0)
            ceil_mode  = self.list_try_get(source_node.args, 4, False)

            if isinstance(padding, tuple):
                if padding[0] == padding[1]:
                    layer.pooling_param.pad.extend([padding[0]])
                else:
                    layer.pooling_param.pad_h = padding[0]
                    layer.pooling_param.pad_w = padding[1]
            else:
                layer.pooling_param.pad = padding

            if isinstance(stride, tuple):
                if stride[0] == stride[1]:
                    layer.pooling_param.stride = stride[0]
                else:
                    layer.pooling_param.stride_h = stride[0]
                    layer.pooling_param.stride_w = stride[1]
            else:
                layer.pooling_param.stride = stride

            if isinstance(kernel_size, tuple):
                if kernel_size[0] == kernel_size[1]:
                    layer.pooling_param.kernel_size = kernel_size[0]
                else:
                    layer.pooling_param.kernel_h = kernel_size[0]
                    layer.pooling_param.kernel_w = kernel_size[1]
            else:
                layer.pooling_param.kernel_size = kernel_size

            layer.pooling_param.ceil_mode = ceil_mode

            bottom_name = self.recursive_find_name(source_node.args[0])
            layer.bottom.append(bottom_name)
            layer.top.append(source_node.name)
            layer.name = source_node.name  

            return layer        

    def rename_max_pool2d_with_indices(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.MAX
        
        kernel_size = source_node.args[1]
        stride = source_node.kwargs['stride']
        padding = source_node.kwargs['padding']
        dilation = source_node.kwargs['dilation']
        ceil_mode = source_node.kwargs['ceil_mode']
        return_indices = source_node.kwargs['return_indices']

        if isinstance(padding, tuple):
            if padding[0] == padding[1]:
                layer.pooling_param.pad = padding[0]
            else:
                layer.pooling_param.pad_h = padding[0]
                layer.pooling_param.pad_w = padding[1]
        else:
            layer.pooling_param.pad = padding             

        if isinstance(stride, tuple):
            if stride[0] == stride[1]:
                layer.pooling_param.stride = stride[0]
            else:
                layer.pooling_param.stride_h = stride[0]
                layer.pooling_param.stride_w = stride[1]
        else:
            layer.pooling_param.stride = stride

        if isinstance(kernel_size, tuple):
            if kernel_size[0] == kernel_size[1]:
                layer.pooling_param.kernel_size = kernel_size[0]
            else:
                layer.pooling_param.kernel_h = kernel_size[0]
                layer.pooling_param.kernel_w = kernel_size[1]
        else:
            layer.pooling_param.kernel_size = kernel_size

        layer.pooling_param.ceil_mode = ceil_mode

        if return_indices:
            bottom_name = self.recursive_find_name(source_node.args[0])
            layer.bottom.append(bottom_name)
            layer.top.append(source_node.name+'_'+str(0))
            layer.top.append(source_node.name+'_'+str(1))
            layer.name = source_node.name
        else:
            self.add_bottom_top(layer, source_node)

        return layer            

    def rename_chunk(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Slice"
        if 'dim' in source_node.kwargs:
            layer.slice_param.axis = source_node.kwargs['dim']
        else:
            layer.slice_param.axis = source_node.args[2]

        sum_ = 0
        for idx in range(len(source_node.meta['tensor_meta'])-1):
            tensor_meta = source_node.meta['tensor_meta'][idx]
            sum_ = sum_ + tensor_meta.shape[layer.slice_param.axis]
            layer.slice_param.slice_point.extend([sum_])

        bottom_name = self.recursive_find_name(source_node.args[0])
        layer.bottom.append(bottom_name)
        for idx in range(len(source_node.meta['tensor_meta'])):
            layer.top.append(source_node.name+'_'+str(idx))
        layer.name = source_node.name

        return layer               

    def rename_split(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Slice"
        layer.slice_param.axis = source_node.kwargs['dim']

        sum_ = 0
        for idx in range(len(source_node.meta['tensor_meta'])-1):
            tensor_meta = source_node.meta['tensor_meta'][idx]
            sum_ = sum_ + tensor_meta.shape[layer.slice_param.axis]
            layer.slice_param.slice_point.extend([sum_])

        bottom_name = self.recursive_find_name(source_node.args[0])
        layer.bottom.append(bottom_name)
        for idx in range(len(source_node.meta['tensor_meta'])):
            layer.top.append(source_node.name+'_'+str(idx))
        layer.name = source_node.name

        return layer                              

    def rename_transpose(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Permute"

        input_dim = len(source_node.args[0].meta['tensor_meta'].shape)
        axes = list(range(input_dim))
        axes[source_node.args[1]], axes[source_node.args[2]] = axes[source_node.args[2]], axes[source_node.args[1]]
        for axe in axes:
            layer.permute_param.order.extend([axe])

        self.add_bottom_top(layer, source_node)

        return layer           

    def rename_prelu(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "PReLU"

        weight = self.model.weight.detach().numpy()

        layer.prelu_param.channel_shared = True
        layer.blobs.extend([as_blob(weight[0])])

        bottom_name = self.recursive_find_name(source_node.args[0])
        layer.bottom.append(bottom_name)
        layer.top.append(source_node.name)
        layer.name = source_node.name   

        return layer