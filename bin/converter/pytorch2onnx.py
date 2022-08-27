import os
import sys
from loguru import logger

import torch
torch.manual_seed(0)
import numpy as np
np.random.seed(0)
import onnxruntime as rt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from core.converter.pytorch.pytorch_onnx_parser import PytorchOnnxParser  # noqa

class Runner(object):
    def __init__(self, name, model, shape, opset_version=13,
                 fuse=False, concrete_args=None):
        self.name = name
        self.model = model
        self.shape = shape
        if isinstance(shape, (tuple, list)) and all(
            isinstance(element, int) for element in shape
        ):
            self.shape = [shape]
        self.opset_version = opset_version
        self.fuse = fuse
        self.concrete_args = concrete_args

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

    def pyotrch_inference(self, generate_onnx=False):
        self.model_file = "tmp/" + self.name
        self.device = torch.device('cpu')
        self.model = self.model.eval().to(self.device)

        self.dummy_input = self.gen_pytorch_input_tensor(self.shape)

        self.pytorch_output = self.model(*self.dummy_input)

        if isinstance(self.pytorch_output, torch.Tensor):
            self.pytorch_output = [self.pytorch_output]

        if generate_onnx:
            torch.onnx.export(self.model, tuple(self.dummy_input),
                              self.name + ".onnx",
                              opset_version=self.opset_version,
                              enable_onnx_checker=False)

    def convert(self):
        pytorch_parser = PytorchOnnxParser(self.model, self.shape,
                                           self.fuse, self.concrete_args)
        pytorch_parser.run()
        pytorch_parser.save(self.model_file)

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
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = \
            rt.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = rt.InferenceSession(self.model_file + ".onnx", sess_options)
        onnx_rt_dict = self.get_onnx_input(sess, self.dummy_input)

        onnx_outname = [output.name for output in sess.get_outputs()]
        self.onnx_output = sess.run(onnx_outname, onnx_rt_dict)

    def check_result(self):
        pytorch_output_list = self.get_tensor_list(self.pytorch_output)
        assert len(pytorch_output_list) == len(self.onnx_output), \
               "pytorch_output: %d vs onnx_output %d" % \
               (len(pytorch_output_list), len(self.onnx_output))

        for idx in range(len(self.onnx_output)):
            np.testing.assert_allclose(
                pytorch_output_list[idx].detach().numpy(),
                self.onnx_output[idx],
                rtol=1e-7,
                atol=1e-3,
            )
        logger.info("accuracy test passed")
