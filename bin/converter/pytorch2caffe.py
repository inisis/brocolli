import os
import re
import sys
from loguru import logger

import torch

torch.manual_seed(0)
import numpy as np

np.random.seed(0)

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

import caffe  # noqa
from core.converter.pytorch.pytorch_caffe_parser import PytorchCaffeParser  # noqa


class Runner(object):
    def __init__(
        self, name, model, shape, opset_version=13, fuse=False, concrete_args=None
    ):
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
        self.device = torch.device("cpu")
        self.model = self.model.eval().to(self.device)

        self.dummy_input = self.gen_pytorch_input_tensor(self.shape)

        self.pytorch_output = self.model(*self.dummy_input)

        if isinstance(self.pytorch_output, torch.Tensor):
            self.pytorch_output = [self.pytorch_output]

        if generate_onnx:
            torch.onnx.export(
                self.model,
                tuple(self.dummy_input),
                self.name + ".onnx",
                opset_version=self.opset_version,
                enable_onnx_checker=False,
            )

    def convert(self):
        pytorch_parser = PytorchCaffeParser(
            self.model, self.shape, self.fuse, self.concrete_args
        )
        pytorch_parser.run()
        pytorch_parser.save(self.model_file)

    def caffe_inference(self):
        prototxt = "tmp/" + self.name + ".prototxt"
        caffemodel = "tmp/" + self.name + ".caffemodel"

        self.net = caffe.Net(prototxt, caffe.TEST, weights=caffemodel)

        if isinstance(self.shape, tuple):
            for idx, _ in enumerate(self.shape):
                img = self.dummy_input[idx].numpy()
                self.net.blobs[self.net.inputs[idx]].data[...] = img
        else:
            img = self.dummy_input[0].numpy()
            self.net.blobs[self.net.inputs[0]].data[...] = img

        self.caffe_output = self.net.forward()

    def check_result(self):
        assert len(self.pytorch_output) == len(
            self.caffe_output
        ), "pytorch_output: %d vs caffe_output %d" % (
            len(self.pytorch_output),
            len(self.caffe_output),
        )

        for idx in range(len(self.caffe_output)):
            np.testing.assert_allclose(
                self.caffe_output[self.net.outputs[idx]],
                self.pytorch_output[idx].detach().numpy(),
                rtol=1e-7,
                atol=1e-3,
            )
        logger.info("accuracy test passed")
