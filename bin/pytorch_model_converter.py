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
        output = self.model(dummy_input)

        pytorch_parser = PytorchParser(self.model, self.shape)
        pytorch_parser.run(model_file)

        prototxt = "tmp/" + self.name + '.prototxt'
        caffemodel = "tmp/" + self.name + '.caffemodel'
        print(prototxt)
        print(caffemodel)
        self.net = caffe.Net(prototxt, caffe.TEST, weights=caffemodel)

        img = np.ones(self.shape)
        self.net.blobs['data'].data[...] = img
        prediction = self.net.forward()

        assert len(output) == len(prediction)

        caffe_outname = net.outputs

        for idx in range(len(output)):
            np.testing.assert_allclose(
                prediction[caffe_outname[idx]].squeeze(),
                output[idx].detach().numpy(),
                atol=1e-03,
            )
        print("accuracy test passed")