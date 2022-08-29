import os
import re
import sys
import numpy as np
np.random.seed(0)
from loguru import logger

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import onnx_layers as ops
from brocolli.converter.pytorch_graph import PytorchGraph

import torch
torch.manual_seed(0)
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval
from torch.fx.node import Node
from torch.fx.graph_module import GraphModule

from onnx import save, helper, checker


class PytorchOnnxParser:
    def __init__(self, model, input_shape, fuse=False, concrete_args=None):
        super(PytorchOnnxParser, self).__init__()
        self.fuse = fuse
        self.model = model.eval()
        self.input_shape = input_shape
        if isinstance(input_shape, (tuple, list)) and all(
            isinstance(element, int) for element in input_shape
        ):
            self.input_shape = [input_shape]
        self.concrete_args = concrete_args

        if isinstance(self.model, GraphModule):
            pass
        elif isinstance(self.model, nn.Module):
            if self.fuse:
                self.fuse_all_conv_bn(self.model)
        else:
            raise Exception("model must be a torch.nn.Module or a torch.fx.GraphModule")

        self.pytorch_graph = PytorchGraph(
            self.model, self.input_shape, self.concrete_args
        )
        self.state_dict = self.pytorch_graph.trace.state_dict()
        self.modules = dict(self.pytorch_graph.trace.named_modules())
        self.in_tensor_value_info = []
        self.nodes = []  # nodes in graph
        self.out_tensor_value_info = []
        self.init_tensor = []

    def convert(self):
        self.gen_ir()

    def fuse_all_conv_bn(self, model):
        stack = []
        for name, module in model.named_children():
            if list(module.named_children()):
                self.fuse_all_conv_bn(module)

            if isinstance(module, nn.BatchNorm2d):
                if not stack:
                    continue
                if isinstance(stack[-1][1], nn.Conv2d):
                    setattr(
                        model, stack[-1][0], fuse_conv_bn_eval(stack[-1][1], module)
                    )
                    setattr(model, name, nn.Identity())
            elif isinstance(module, nn.BatchNorm1d):
                if not stack:
                    continue
                if isinstance(stack[-1][1], nn.Linear):
                    setattr(
                        model, stack[-1][0], fuse_linear_bn_eval(stack[-1][1], module)
                    )
                    setattr(model, name, nn.Identity())
            else:
                stack.append((name, module))

    def gen_ir(self):
        for node in self.pytorch_graph.nodes:
            if node.op == "placeholder":
                input_layer = ops.InputLayer(node)
                self.node_post_process(input_layer)
            elif node.op == "call_module":
                module = self.modules[node.target]
                if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                    conv_layer = ops.ConvLayer(node, module)
                    self.node_post_process(conv_layer)
                elif isinstance(module, nn.BatchNorm2d):
                    batchnorm_layer = ops.BatchNormLayer(node, module)
                    self.node_post_process(batchnorm_layer)
                elif isinstance(module, nn.ReLU):
                    relu_layer = ops.ReluLayer(node, module)
                    self.node_post_process(relu_layer)
                elif isinstance(module, (nn.MaxPool1d, nn.MaxPool2d)):
                    pooling_layer = ops.PoolingLayer(node, module)
                    self.node_post_process(pooling_layer)
                elif isinstance(module, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d)):
                    pooling_layer = ops.PoolingLayer(node, module)
                    self.node_post_process(pooling_layer)
                elif isinstance(module, nn.Linear):
                    linear_layer = ops.LinearLayer(node, module)
                    self.node_post_process(linear_layer)
                elif isinstance(module, nn.Dropout):
                    dropout_layer = ops.DropoutLayer(node, module)
                    self.node_post_process(dropout_layer)
                elif isinstance(module, nn.ReLU6):
                    relu6_layer = ops.Relu6Layer(node, module)
                    self.node_post_process(relu6_layer)
                elif isinstance(module, nn.Hardswish):
                    hardsiwsh_layer = ops.HardswishLayer(node, module)
                    self.node_post_process(hardsiwsh_layer)
                elif isinstance(module, nn.Hardsigmoid):
                    hardsigmoid_layer = ops.HardsigmoidLayer(node, module)
                    self.node_post_process(hardsigmoid_layer)
                elif isinstance(module, nn.Identity):
                    identity_layer = ops.IdentityLayer(node, module)
                    self.node_post_process(identity_layer)
                elif isinstance(module, (nn.AvgPool1d, nn.AvgPool2d)):
                    avgpool_layer = ops.AvgPoolLayer(node, module)
                    self.node_post_process(avgpool_layer)
                elif isinstance(module, nn.Upsample):
                    upsample_layer = ops.UpsampleLayer(node, module)
                    self.node_post_process(upsample_layer)
                elif isinstance(module, nn.PReLU):
                    prelu_layer = ops.PReluLayer(node, module)
                    self.node_post_process(prelu_layer)
                elif isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d)):
                    conv_transpose_layer = ops.ConvTransposeLayer(node, module)
                    self.node_post_process(conv_transpose_layer)
                elif isinstance(module, nn.LSTM):
                    lstm_layer = ops.LSTMLayer(node, module)
                    self.node_post_process(lstm_layer)
                elif isinstance(module, nn.RNN):
                    rnn_layer = ops.RNNLayer(node, module)
                    self.node_post_process(rnn_layer)
                elif isinstance(module, nn.GRU):
                    gru_layer = ops.GRULayer(node, module)
                    self.node_post_process(gru_layer)
                elif isinstance(module, nn.Flatten):
                    flatten_layer = ops.FlattenLayer(node, module)
                    self.node_post_process(flatten_layer)
                elif isinstance(module, nn.LeakyReLU):
                    leakyrelu_layer = ops.LeakyReluLayer(node, module)
                    self.node_post_process(leakyrelu_layer)
                elif isinstance(
                    module, (nn.ConstantPad1d, nn.ConstantPad2d, nn.ConstantPad3d)
                ):
                    pad_layer = ops.PadLayer(node, module)
                    self.node_post_process(pad_layer)
                elif isinstance(module, (nn.ReflectionPad1d, nn.ReflectionPad2d)):
                    pad_layer = ops.PadLayer(node, module)
                    self.node_post_process(pad_layer)
                elif isinstance(module, (nn.ReplicationPad1d, nn.ReplicationPad2d)):
                    pad_layer = ops.PadLayer(node, module)
                    self.node_post_process(pad_layer)
                elif isinstance(module, nn.SELU):
                    selu_layer = ops.SeluLayer(node, module)
                    self.node_post_process(selu_layer)
                elif isinstance(module, nn.ELU):
                    elu_layer = ops.EluLayer(node, module)
                    self.node_post_process(elu_layer)
                elif isinstance(module, nn.Sigmoid):
                    sigmoid_layer = ops.SigmoidLayer(node, module)
                    self.node_post_process(sigmoid_layer)
                elif isinstance(module, nn.Softmax):
                    softmax_layer = ops.SoftmaxLayer(node, module)
                    self.node_post_process(softmax_layer)
                elif isinstance(module, nn.Softplus):
                    layer = ops.SoftplusLayer(node, module)
                    self.node_post_process(layer)
                else:
                    raise NotImplementedError("module %s is not implemented" % (module))
            elif node.op == "call_function":
                function_name = re.findall(
                    r"(?:function|method) ([a-z|_|0-9]+.*?)", str(node.target)
                )[0]
                if function_name == "relu":
                    relu_layer = ops.ReluFunc(node)
                    self.node_post_process(relu_layer)
                elif function_name == "add":
                    add_layer = ops.AddFunc(node)
                    self.node_post_process(add_layer)
                elif function_name == "flatten":
                    flatten_layer = ops.FlattenFunc(node)
                    self.node_post_process(flatten_layer)
                elif function_name == "cat":
                    concat_layer = ops.ConcatFunc(node)
                    self.node_post_process(concat_layer)
                elif (
                    function_name == "adaptive_avg_pool2d"
                    or function_name == "adaptive_avg_pool1d"
                ):
                    pooling_layer = ops.PoolingFunc(node)
                    self.node_post_process(pooling_layer)
                elif function_name == "hardsigmoid":
                    hardsigmoid_layer = ops.HardsigmoidFunc(node)
                    self.node_post_process(hardsigmoid_layer)
                elif function_name == "mul":
                    mul_layer = ops.MulFunc(node)
                    self.node_post_process(mul_layer)
                elif function_name == "getitem":
                    if isinstance(node.args[1], tuple):
                        if all(
                            isinstance(function, slice) for function in node.args[1]
                        ):
                            params_slices = []
                            for idx, function in enumerate(node.args[1]):
                                if (
                                    function.start is None
                                    and function.stop is None
                                    and function.step is None
                                ):
                                    continue
                                else:
                                    start_ = (
                                        function.start
                                        if function.start is not None
                                        else 0
                                    )
                                    end_ = (
                                        function.stop
                                        if function.stop is not None
                                        else 1
                                    )  # maybe a bug
                                    axes_ = idx
                                    step_ = (
                                        function.step
                                        if function.step is not None
                                        else 1
                                    )

                                    params_slice = [
                                        np.array([start_]),
                                        np.array([end_]),
                                        np.array([axes_]),
                                        np.array([step_]),
                                    ]
                                    params_slices.append(params_slice)

                            if len(params_slices) == 1:
                                slice_layer = ops.SliceFunc(node, auto_gen=False)
                                slice_layer.add_bottom_top()
                                slice_layer.generate_node(params=params_slices[0])
                                self.node_post_process(slice_layer)
                        else:
                            raise
                    pass
                elif function_name == "floordiv":
                    pass
                elif function_name == "transpose":
                    transpose_layer = ops.TransposeFunc(node)
                    self.node_post_process(transpose_layer)
                elif function_name == "prelu":
                    prelu_layer = ops.PReluFunc(node, auto_gen=False)
                    prelu_layer.add_bottom_top()
                    params_prelu = self.model.weight.detach().numpy()
                    prelu_layer.generate_params(params_prelu)
                    prelu_layer.generate_node()
                    self.node_post_process(prelu_layer)
                elif function_name == "hardtanh":
                    clip_layer = ops.ClipFunc(node, auto_gen=False)
                    clip_layer.add_bottom_top()
                    params_clip = [
                        np.array(node.kwargs["min_val"]),
                        np.array(node.kwargs["max_val"]),
                    ]
                    clip_layer.generate_params(params_clip)
                    clip_layer.generate_node()
                    self.node_post_process(clip_layer)
                elif function_name == "leaky_relu":
                    leakyrelu_layer = ops.LeakyReluFunc(node)
                    self.node_post_process(leakyrelu_layer)
                elif function_name == "sigmoid":
                    sigmoid_layer = ops.SigmoidFunc(node)
                    self.node_post_process(sigmoid_layer)
                elif function_name == "softmax":
                    softmax_layer = ops.SoftmaxFunc(node)
                    self.node_post_process(softmax_layer)
                elif function_name == "hardswish":
                    hardswish_layer = ops.HardswishFunc(node)
                    self.node_post_process(hardswish_layer)
                elif function_name == "conv2d":
                    conv_layer = ops.ConvFunc(node, auto_gen=False)
                    conv_layer.add_bottom_top()
                    weight = self.model.weight.detach().numpy()
                    bias = self.model.bias
                    if bias is not None:
                        params_conv = [weight, bias.detach().numpy()]
                    else:
                        params_conv = [weight]
                    conv_layer.generate_params(params_conv)
                    conv_layer.generate_node()
                    self.node_post_process(conv_layer)
                elif function_name == "linear":
                    gemm_layer = ops.GemmFunc(node, auto_gen=False)

                    gemm_layer.add_bottom_top()
                    weight = self.model.weight.detach().numpy()
                    bias = self.model.bias
                    if bias is not None:
                        params_conv = [weight, bias.detach().numpy()]
                    else:
                        params_conv = [weight]
                    gemm_layer.generate_params(params_conv)
                    gemm_layer.generate_node()
                    self.node_post_process(gemm_layer)
                elif function_name == "avg_pool2d" or function_name == "avg_pool1d":
                    avgpool_layer = ops.AvgPoolFunc(node)
                    self.node_post_process(avgpool_layer)
                elif function_name == "chunk":
                    chunk_layer = ops.ChunkFunc(node)
                    self.node_post_process(chunk_layer)
                elif function_name == "split":
                    split_layer = ops.SplitFunc(node)
                    self.node_post_process(split_layer)
                elif function_name == "getattr":
                    pass
                elif function_name == "boolean_dispatch":
                    if "max_pool2d" in node.name or "max_pool1d" in node.name:
                        pooling_layer = ops.PoolingFunc(node)
                        self.node_post_process(pooling_layer)
                    else:
                        raise NotImplementedError(
                            "function %s is not implemented" % (node.name)
                        )
                elif function_name == "relu6":
                    relu6_layer = ops.Relu6Func(node)
                    self.node_post_process(relu6_layer)
                elif function_name == "max":
                    max_layer = ops.MaxFunc(node)
                    self.node_post_process(max_layer)
                elif function_name == "exp":
                    exp_layer = ops.ExpFunc(node)
                    self.node_post_process(exp_layer)
                elif function_name == "log":
                    log_layer = ops.LogFunc(node)
                    self.node_post_process(log_layer)
                elif function_name == "min":
                    min_layer = ops.MinFunc(node)
                    self.node_post_process(min_layer)
                elif function_name == "elu":
                    elu_layer = ops.EluFunc(node)
                    self.node_post_process(elu_layer)
                elif function_name == "selu":
                    selu_layer = ops.SeluFunc(node)
                    self.node_post_process(selu_layer)
                elif function_name == "abs":
                    abs_layer = ops.AbsFunc(node)
                    self.node_post_process(abs_layer)
                elif function_name == "sqrt":
                    sqrt_layer = ops.SqrtFunc(node)
                    self.node_post_process(sqrt_layer)
                elif function_name == "pow":
                    pow_layer = ops.PowerFunc(node)
                    self.node_post_process(pow_layer)
                elif function_name == "sin":
                    sin_layer = ops.SineFunc(node)
                    self.node_post_process(sin_layer)
                elif function_name == "cos":
                    cos_layer = ops.CosineFunc(node)
                    self.node_post_process(cos_layer)
                elif function_name == "celu":
                    celu_layer = ops.CeluFunc(node)
                    self.node_post_process(celu_layer)
                elif function_name == "sum":
                    sum_layer = ops.SumFunc(node)
                    self.node_post_process(sum_layer)
                elif function_name == "neg":
                    neg_layer = ops.NegFunc(node)
                    self.node_post_process(neg_layer)
                elif function_name == "tanh":
                    tanh_layer = ops.TanhFunc(node)
                    self.node_post_process(tanh_layer)
                elif function_name == "mean":
                    mean_layer = ops.MeanFunc(node)
                    self.node_post_process(mean_layer)
                elif function_name == "sub":
                    sub_layer = ops.SubFunc(node)
                    self.node_post_process(sub_layer)
                elif function_name == "div":
                    div_layer = ops.DivFunc(node)
                    self.node_post_process(div_layer)
                elif function_name == "matmul":
                    matmul_layer = ops.MatmulFunc(node)
                    self.node_post_process(matmul_layer)
                elif function_name == "softplus":
                    softplus_layer = ops.SoftplusFunc(node)
                    self.node_post_process(softplus_layer)
                elif function_name == "interpolate":
                    upsample_layer = ops.UpsampleFunc(node)
                    self.node_post_process(upsample_layer)
                elif function_name == "_pad":
                    pad_layer = ops.PadFunc(node)
                    self.node_post_process(pad_layer)
                elif function_name == "tile":
                    tile_layer = ops.TileFunc(node)
                    self.node_post_process(tile_layer)
                else:
                    raise NotImplementedError(
                        "function %s is not implemented" % (function_name)
                    )
            elif node.op == "call_method":
                if str(node.target) == "size":
                    pass
                elif str(node.target) == "view":
                    reshape_layer = ops.ReshapeFunc(node)
                    self.node_post_process(reshape_layer)
                elif str(node.target) == "reshape":
                    reshape_layer = ops.ReshapeFunc(node)
                    self.node_post_process(reshape_layer)
                elif str(node.target) == "contiguous":
                    pass
                elif str(node.target) == "chunk":
                    chunk_layer = ops.ChunkFunc(node)
                    self.node_post_process(chunk_layer)
                elif str(node.target) == "mean":
                    mean_layer = ops.MeanFunc(node)
                    self.node_post_process(mean_layer)
                elif str(node.target) == "permute":
                    permute_layer = ops.PermuteFunc(node)
                    self.node_post_process(permute_layer)
                elif str(node.target) == "sigmoid":
                    sigmoid_layer = ops.SigmoidFunc(node)
                    self.node_post_process(sigmoid_layer)
                elif str(node.target) == "tanh":
                    tanh_layer = ops.TanhFunc(node)
                    self.node_post_process(tanh_layer)
                elif str(node.target) == "repeat":
                    tile_layer = ops.TileFunc(node)
                    self.node_post_process(tile_layer)
                elif str(node.target) == "unsqueeze":
                    unsqueeze_layer = ops.UnsqueezeFunc(node)
                    self.node_post_process(unsqueeze_layer)
                elif str(node.target) == "squeeze":
                    squeeze_layer = ops.SqueezeFunc(node)
                    self.node_post_process(squeeze_layer)
                elif str(node.target) == "cos":
                    cos_layer = ops.CosineFunc(node)
                    self.node_post_process(cos_layer)
                elif str(node.target) == "pow":
                    pow_layer = ops.PowerFunc(node)
                    self.node_post_process(pow_layer)
                elif str(node.target) == "sin":
                    sin_layer = ops.SineFunc(node)
                    self.node_post_process(sin_layer)
                elif str(node.target) == "abs":
                    abs_layer = ops.AbsFunc(node)
                    self.node_post_process(abs_layer)
                elif str(node.target) == "log":
                    log_layer = ops.LogFunc(node)
                    self.node_post_process(log_layer)
                elif str(node.target) == "sqrt":
                    sqrt_layer = ops.SqrtFunc(node)
                    self.node_post_process(sqrt_layer)
                else:
                    raise NotImplementedError(
                        "method %s is not implemented" % (str(node.target))
                    )
            elif node.op == "output":
                output_layer = ops.OutputLayer(node)
                self.node_post_process(output_layer)
            elif node.op == "get_attr":
                pass
            else:
                raise NotImplementedError("op type %s is not implemented" % (node.op))

    def save(self, dest_path):
        self.dest_path = dest_path
        graph_def = helper.make_graph(
            self.nodes,
            dest_path,
            self.in_tensor_value_info,
            self.out_tensor_value_info,
            self.init_tensor,
        )
        self.model_def = helper.make_model(graph_def, producer_name="pytorch")
        self.freeze()
        checker.check_model(self.model_def)
        logger.info("onnx model conversion completed")
        save(self.model_def, dest_path)
        logger.info("onnx model saved to {}".format(dest_path))

    def check_result(self):
        self.pyotrch_inference()
        self.onnx_inference()
        pytorch_output_list = self.get_tensor_list(self.pytorch_output)
        assert len(pytorch_output_list) == len(
            self.onnx_output
        ), "pytorch_output: %d vs onnx_output %d" % (
            len(pytorch_output_list),
            len(self.onnx_output),
        )

        for idx in range(len(self.onnx_output)):
            np.testing.assert_allclose(
                pytorch_output_list[idx].detach().numpy(),
                self.onnx_output[idx],
                rtol=1e-7,
                atol=1e-3,
            )
        logger.info("accuracy test passed")

    def gen_pytorch_input_tensor(self, shapes):
        input_tensor = []
        for shape in shapes:
            if isinstance(shape, (tuple, list)):
                if all(isinstance(element, int) for element in shape):
                    input_tensor.append(torch.rand(shape).to(torch.float32))
                else:
                    input_tensor.append(self.gen_pytorch_input_tensor(shape))
            else:
                input_tensor.append(torch.rand(shape).to(torch.float32))

        return input_tensor

    def pyotrch_inference(self):
        self.dummy_input = self.gen_pytorch_input_tensor(self.input_shape)

        self.pytorch_output = self.model(*self.dummy_input)

        if isinstance(self.pytorch_output, torch.Tensor):
            self.pytorch_output = [self.pytorch_output]

    def get_tensor_list(self, dummy_inputs):
        tensor_list = []
        for dummy_input in dummy_inputs:
            if isinstance(dummy_input, torch.Tensor):
                tensor_list.append(dummy_input)
            else:
                tensor_list.extend(self.get_tensor_list(dummy_input))

        return tensor_list

    def get_onnx_input(self, sess, dummy_inputs):
        dummy_input_list = self.get_tensor_list(dummy_inputs)

        onnx_rt_dict = {}
        for idx in range(len(dummy_input_list)):
            img = dummy_input_list[idx].numpy()
            onnx_rt_dict[sess.get_inputs()[idx].name] = img

        return onnx_rt_dict

    def onnx_inference(self):
        import onnxruntime as rt
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = (
            rt.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        sess = rt.InferenceSession(self.dest_path, sess_options)
        onnx_rt_dict = self.get_onnx_input(sess, self.dummy_input)

        onnx_outname = [output.name for output in sess.get_outputs()]
        self.onnx_output = sess.run(onnx_outname, onnx_rt_dict)

    def node_post_process(self, onnx_layer):
        if onnx_layer._node:
            self.nodes.extend(onnx_layer._node)
        self.in_tensor_value_info.extend(onnx_layer._in_tensor_value_info)
        self.out_tensor_value_info.extend(onnx_layer._out_tensor_value_info)
        self.init_tensor.extend(onnx_layer._init_tensor)

    def freeze(self):
        logger.info("removing not constant initializers from model")
        inputs = self.model_def.graph.input
        name_to_input = {}
        for input in inputs:
            name_to_input[input.name] = input

        for initializer in self.model_def.graph.initializer:
            if initializer.name in name_to_input:
                inputs.remove(name_to_input[initializer.name])
