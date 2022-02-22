import os
import re
import sys
import argparse
import json

import torch
torch.manual_seed(0)
import numpy as np
np.random.seed(0)

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import caffe  # noqa
from converter.pytorch.pytorch_parser import PytorchParser  # noqa

class Runner(object):
    def __init__(self, name, model, shape, opset_version, fuse=False):
        self.name = name
        self.model = model
        self.shape = shape
        self.opset_version = opset_version
        self.fuse = fuse

    def pyotrch_inference(self, generate_onnx=False):
        self.model_file = "tmp/" + self.name
        self.device = torch.device('cpu')
        self.model = self.model.eval().to(self.device)
        if isinstance(self.shape, tuple):
            dummy_input = []
            for each in self.shape:
                dummy = torch.ones(each).to(torch.float32)
                dummy_input.append(dummy)
        else:
            dummy_input = torch.ones(self.shape).to(torch.float32)

        self.pytorch_output = self.model(dummy_input)
 
        if generate_onnx:
            torch.onnx.export(self.model, dummy_input, self.name + ".onnx", opset_version=self.opset_version, enable_onnx_checker=False)
        
    def convert(self, export_mode=False):
        self.model.export_mode = export_mode
        pytorch_parser = PytorchParser(self.model, self.shape, self.opset_version, self.fuse)
        pytorch_parser.run(self.model_file)

    def caffe_inference(self):
        prototxt = "tmp/" + self.name + '.prototxt'
        caffemodel = "tmp/" + self.name + '.caffemodel'

        self.net = caffe.Net(prototxt, caffe.TEST, weights=caffemodel)

        if isinstance(self.shape, tuple):
            for idx, each in enumerate(self.shape):
                img = np.ones(each)
                self.net.blobs['data_' + str(idx)].data[...] = img
        else:
            img = np.ones(self.shape)
            self.net.blobs['data'].data[...] = img

        self.caffe_output = self.net.forward()

    def check_result(self):
        assert len(self.pytorch_output) == len(self.caffe_output)

        caffe_outname = self.net.outputs
        caffe_outname = sorted(caffe_outname, key=lambda x: re.findall(r'\d+', x)[-1])

        for idx in range(len(self.caffe_output)):
            np.testing.assert_allclose(
                self.caffe_output[caffe_outname[idx]].flatten(),
                self.pytorch_output[idx].detach().numpy().flatten(),
                rtol=1e-3,
                atol=0, # inception will produce large outputs, but low relative error
            )
        print("accuracy test passed")