import os
import sys
import argparse
import json

import torch
torch.set_printoptions(precision=10)
import numpy as np
from torchsummary import summary

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import caffe  # noqa
from converter.pytorch.pytorch_parser import PytorchParser  # noqa

class Runner(object):
    def __init__(self, name, model, shape):
        self.name = name
        self.model = model
        self.shape = shape

    def inference(self):
        model_file = "tmp/" + self.name
        device = torch.device('cpu')
        self.model = self.model.eval().to(device)

        dummy_input = torch.ones(self.shape).to(device)
        pytorch_output = self.model(dummy_input)

        pytorch_parser = PytorchParser(self.model, self.shape)
        pytorch_parser.run(model_file)

        prototxt = "tmp/" + self.name + '.prototxt'
        caffemodel = "tmp/" + self.name + '.caffemodel'

        self.net = caffe.Net(prototxt, caffe.TEST, weights=caffemodel)

        img = np.ones(self.shape)
        self.net.blobs['data'].data[...] = img
        caffe_output = self.net.forward()

        assert len(pytorch_output) == len(caffe_output)

        caffe_outname = self.net.outputs
        for idx in range(len(caffe_output)):
            np.testing.assert_allclose(
                caffe_output[caffe_outname[idx]].squeeze(),
                pytorch_output[idx].detach().numpy(),
                rtol=1e-03,
            )
        print("accuracy test passed")