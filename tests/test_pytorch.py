from __future__ import absolute_import
from __future__ import print_function

import os
import sys

import torch
import caffe
import numpy as np
import torch.nn as nn
from torchsummary import summary
from converter.resnet import *
from converter.ssd import *

from converter.pytorch.pytorch_parser import PytorchParser

# model_file = "model/resnet.pkl"
#
# model = resnet18(pretrained=True)
#
# model.eval()
#
# # model.bn1.weight.data.fill_(1)
# # model.bn1.bias.data.fill_(0)
#
# dummy_input = torch.autograd.Variable(torch.ones([1, 3, 224, 224]), requires_grad=False)
#
# outputs = []
# def hook(module, input, output):
#     outputs.append(output)
#
# model.maxpool.register_forward_hook(hook)
#
# output = model(dummy_input)
#
# print(outputs)
#
# torch.save(model, 'model/resnet.pkl')
#
# model = torch.load(model_file, map_location='cpu')
#
# print(output)

model_file = "model/VOC.pkl"
#
device = torch.device("cpu") # PyTorch v0.4.0
# ssd_net = build_ssd('train', 300, 21)
# ssd_net.to(device)
#
# print(ssd_net)
# net = ssd_net
# ssd_net.load_weights("model/VOC.pkl")
net = torch.load("model/VOC.pkl")
net.eval()

dummy_input = torch.autograd.Variable(torch.ones([1, 3, 300, 300]), requires_grad=False)

outputs = []
def hook(module, input, output):
    outputs.append(output)

# net.L2Norm.register_forward_hook(hook)
net.vgg_back[0].register_forward_hook(hook)

net.to(device)
output = net(dummy_input)

device = torch.device("cuda") # PyTorch v0.4.0
summary(net.to(device), (3, 300, 300))

# torch.save(net, 'model/VOC.pkl')

parser = PytorchParser(model_file, [3, 300, 300])
#
parser.run(model_file)

Model_FILE = model_file + '.prototxt'

PRETRAINED = model_file + '.caffemodel'

IMAGE_FILE = 'data/avatar.jpg'

input_image = caffe.io.load_image(IMAGE_FILE)

net = caffe.Classifier(Model_FILE, PRETRAINED)

caffe.set_mode_cpu()

# caffe.set_mode_gpu()

img = np.ones((3, 300, 300))

input_data = net.blobs["data"].data[...]

net.blobs['data'].data[...] = img

prediction = net.forward()

# W = net.params['ResNetnBatchNorm2dnbn1n104_bn'][0].data[...]
# b = net.params['ResNetnBatchNorm2dnbn1n104_bn'][1].data[...]
# b2 = net.params['ResNetnBatchNorm2dnbn1n104_bn'][2].data[...]
#
# W = net.params['ResNetnBatchNorm2dnbn1n104_scale'][0].data[...]
# b = net.params['ResNetnBatchNorm2dnbn1n104_scale'][1].data[...]

def print_CNNfeaturemap(net, output_dir):
    params = list(net.blobs.keys())
    print (params)
    for pr in params[0:]:
        print (pr)
        res = net.blobs[pr].data[...]
        pr = pr.replace('/', '_')
        print (res.shape)
        for index in range(0,res.shape[0]):
           if len(res.shape) == 4:
              filename = os.path.join(output_dir, "%s_output%d_%d_%d_%d_caffe.linear.float"%(pr,index,res.shape[1],res.shape[2],res.shape[3]))
           elif len(res.shape) == 3:
              filename = os.path.join(output_dir, "%s_output%d_%d_%d_caffe.linear.float"%(pr, index,res.shape[1],res.shape[2]))
           elif len(res.shape) == 2:
              filename = os.path.join(output_dir, "%s_output%d_%d_caffe.linear.float"%(pr,index,res.shape[1]))
           elif len(res.shape) == 1:
              filename = os.path.join(output_dir, "%s_output%d_caffe.linear.float"%(pr,index))
           f = open(filename, 'wb')

           np.savetxt(f, list(res.reshape(-1, 1)))

print_CNNfeaturemap(net, "model/cnn_result")