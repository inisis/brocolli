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

import torchvision.models as models

model_file = "pytorch_model/best.pth"

device = torch.device('cpu')
net = models.googlenet(pretrained=False)
net.eval()
torch.save(net, model_file)

dummy_input = torch.ones([1, 3, 224, 224]).to(device)
net.to(device)

output = net(dummy_input)

pytorch_parser = PytorchParser(model_file, [3, 224, 224])

pytorch_parser.run(model_file)

prototxt = model_file + '.prototxt'
caffemodel = model_file + '.caffemodel'
net = caffe.Classifier(prototxt, caffemodel)
caffe.set_mode_cpu()

img = np.ones((3, 224, 224))
input_data = net.blobs["data"].data[...]
net.blobs['data'].data[...] = img
prediction = net.forward()

print(output)
print(prediction)

assert len(output) == len(prediction)

caffe_outname = net.outputs

for idx in range(len(output)):
    np.testing.assert_allclose(
        prediction[caffe_outname[idx]].squeeze(),
        output[idx].detach().numpy(),
        atol=1e-03,
    )
print("accuracy test passed")