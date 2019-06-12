from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import argparse
import json
from easydict import EasyDict as edict

import torch
import numpy as np
from torchsummary import summary

sys.path.append('/home/yaojin/github/caffe/python')
sys.path.append('/home/yaojin/github/caffe/python/caffe')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import caffe  # noqa
from converter.pytorch.pytorch_parser import PytorchParser  # noqa
from model.classifier import Classifier  # noqa

parser = argparse.ArgumentParser(description='test converter')
parser.add_argument('model_path', default=None, metavar='MODEL_PATH', type=str,
                    help="Path to the trained models")
args = parser.parse_args()

with open(args.model_path+'cfg.json') as f:
    cfg = edict(json.load(f))

model_file = "model/best.pth"
device = torch.device('cpu')  # PyTorch v0.4.0
net = Classifier(cfg)
ckpt = torch.load("model/best.ckpt")
net.load_state_dict(ckpt['state_dict'], strict=False)
torch.save(net, model_file)

net.eval()

dummy_input = torch.ones([1, 3, 1024, 1024])
outputs = []

def hook(module, input, output):
        #print(output.data)
        outputs.append(output)

def PrintTorch(net,outputdir="model/torch_result"):

    for name,moudel in net.named_children():
        print(name)
        ff= open(os.path.join(outputdir,name),'wb')
        handle = moudel.register_forward_hook(hook) 
        net.to(device)                                                                                   
        _ = net(dummy_input)
        handle.remove()
        out=outputs[0].cpu().detach().numpy()
        outputs.pop()
        np.savetxt(ff,list(out.reshape(-1,1)))
        ff.close()
#PrintTorch(net)

net.to(device)
output = net(dummy_input)
device = torch.device("cuda")  # PyTorch v0.4.0
summary(net.to(device), (3, 1024, 1024))

pytorch_parser = PytorchParser(model_file, [3, 1024, 1024])
#
pytorch_parser.run(model_file)

Model_FILE = model_file + '.prototxt'

PRETRAINED = model_file + '.caffemodel'

net = caffe.Classifier(Model_FILE, PRETRAINED)

caffe.set_mode_cpu()

# caffe.set_mode_gpu()

img = np.ones((3, 1024, 1024))

input_data = net.blobs["data"].data[...]

net.blobs['data'].data[...] = img

prediction = net.forward()

print(output)
print(prediction)


def print_CNNfeaturemap(net, output_dir):
    params = list(net.blobs.keys())
    for pr in params[0:]:
        res = net.blobs[pr].data[...]
        pr = pr.replace('/', '_')
        for index in range(0, res.shape[0]):
            if len(res.shape) == 4:
                filename = os.path.join(output_dir,
                                        "%s_output%d_%d_%d_%d_caffe.linear.float"  # noqa
                                        % (pr, index, res.shape[1],
                                           res.shape[2], res.shape[3]))
            elif len(res.shape) == 3:
                filename = os.path.join(output_dir,
                                        "%s_output%d_%d_%d_caffe.linear.float"
                                        % (pr, index, res.shape[1],
                                           res.shape[2]))
            elif len(res.shape) == 2:
                filename = os.path.join(output_dir,
                                        "%s_output%d_%d_caffe.linear.float"
                                        % (pr, index, res.shape[1]))
            elif len(res.shape) == 1:
                filename = os.path.join(output_dir,
                                        "%s_output%d_caffe.linear.float"
                                        % (pr, index))
            f = open(filename, 'wb')
            #print(res.shape)
            np.savetxt(f, list(res.reshape(-1, 1)))
            f.close()

print_CNNfeaturemap(net, "model/cnn_result")
