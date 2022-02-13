import os
os.environ[
    "GLOG_minloglevel"
] = "3"  # 0 - debug 1 - info (still a LOT of outputs) 2 - warnings 3 - errors
import argparse
import numpy as np

import caffe
import pytest
import warnings

import torchvision.models as models

from bin.pytorch_model_converter import Runner


def test_pooling(shape = [1, 1, 32, 32], opset_version=13):
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.pool1 = nn.MaxPool2d(2, stride=2, return_indices=True)
            self.unpool1 = nn.MaxUnpool2d(2, stride=2)

        def forward(self, x):
            x = self.conv1(x)
            x, indice = self.pool1(x)
            x = self.unpool1(x, indice)
            return x
    
    net = Net()
    runner = Runner("pooling", net, shape, opset_version)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pytest.main(['-p', 'no:warnings', '-v', 'test/test_ops/py'])
