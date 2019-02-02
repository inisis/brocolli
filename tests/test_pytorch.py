from __future__ import absolute_import
from __future__ import print_function

import os
import sys

import torch
import caffe
import numpy as np

from converter.pytorch.pytorch_parser import PytorchParser

model_file = "model/pnet_epoch_model_10.pkl"

parser = PytorchParser("model/pnet_epoch_model_10.pkl", [3, 12, 12])

parser.run(model_file)

model = torch.load(model_file, map_location='cpu')

dummy_input = torch.autograd.Variable(torch.ones([1, 3, 12, 12]), requires_grad=False)

output = model(dummy_input)

print(output)

Model_FILE = model_file + '.prototxt'

PRETRAINED = model_file + '.caffemodel'

IMAGE_FILE = 'data/avatar.jpg'

input_image = caffe.io.load_image(IMAGE_FILE)

net = caffe.Classifier(Model_FILE, PRETRAINED)

caffe.set_mode_cpu()

# caffe.set_mode_gpu()

img = np.ones((3, 12, 12))

input_data = net.blobs["data"].data[...]

net.blobs['data'].data[...] = img

prediction = net.forward()

print('predicted calss: ', prediction[0].argmax())