import os

os.environ[
    "GLOG_minloglevel"
] = "3"  # 0 - debug 1 - info (still a LOT of outputs) 2 - warnings 3 - errors
import argparse
import numpy as np

import caffe
caffe.set_mode_cpu()
import torch
import pytest
import warnings

import torchvision.models as models
from converter.pytorch.pytorch_parser import PytorchParser


def test_resnet18(shape = [1, 3, 224, 224]):
    model_file = "tmp/best.pth"
    
    device = torch.device('cpu')
    net = models.resnet18(pretrained=False).eval().to(device)
    torch.save(net, model_file)
    dummy_input = torch.ones(shape).to(device)
    output = net(dummy_input)

    pytorch_parser = PytorchParser(model_file, shape)
    pytorch_parser.run(model_file)

    prototxt = model_file + '.prototxt'
    caffemodel = model_file + '.caffemodel'
    net = caffe.Classifier(prototxt, caffemodel)

    img = np.ones(shape)
    input_data = net.blobs["data"].data[...]
    net.blobs['data'].data[...] = img
    prediction = net.forward()

    assert len(output) == len(prediction)

    caffe_outname = net.outputs

    for idx in range(len(output)):
        np.testing.assert_allclose(
            prediction[caffe_outname[idx]].squeeze(),
            output[idx].detach().numpy(),
            atol=1e-03,
        )
    print("accuracy test passed")
        
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pytest.main(['-p', 'no:warnings', '-v', 'tests'])
