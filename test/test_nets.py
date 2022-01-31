import os
os.environ[
    "GLOG_minloglevel"
] = "3"  # 0 - debug 1 - info (still a LOT of outputs) 2 - warnings 3 - errors
import argparse
import numpy as np

import caffe
import torch
import pytest
import warnings

import torchvision.models as models



import os

os.environ[
    "GLOG_minloglevel"
] = "3"  # 0 - debug 1 - info (still a LOT of outputs) 2 - warnings 3 - errors

from bin.pytorch_model_converter import Runner

def test_alexnet(shape = [1, 3, 224, 224]):
    net = models.alexnet(pretrained=False)
    runner = Runner("alexnet", net, shape)
    runner.inference()

def test_resnet18(shape = [1, 3, 224, 224]):
    net = models.resnet18(pretrained=False)
    runner = Runner("resnet18", net, shape)
    runner.inference()

def test_squeezenet(shape = [1, 3, 227, 227]):
    net = models.squeezenet1_0(pretrained=False)
    runner = Runner("squeezenet", net, shape)
    runner.inference()

def test_googlenet(shape = [1, 3, 224, 224]):
    net = models.googlenet(pretrained=False)
    runner = Runner("googlenet", net, shape)
    runner.inference()

# def test_shufflenet(shape = [1, 3, 224, 224]):
#     net = models.shufflenet_v2_x1_0(pretrained=False)
#     runner = Runner("shufflenet", net, shape)
#     runner.inference()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pytest.main(['-p', 'no:warnings', '-v', 'test'])
