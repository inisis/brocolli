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

def test_alexnet(shape = [1, 3, 224, 224], opset_version=9):
    net = models.alexnet(pretrained=False)
    runner = Runner("alexnet", net, shape, opset_version)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_resnet18(shape = [1, 3, 224, 224], opset_version=9):
    net = models.resnet18(pretrained=False)
    runner = Runner("resnet18", net, shape, opset_version)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_squeezenet(shape = [1, 3, 227, 227], opset_version=9):
    net = models.squeezenet1_0(pretrained=False)
    runner = Runner("squeezenet", net, shape, opset_version)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_googlenet(shape = [1, 3, 224, 224], opset_version=9):
    net = models.googlenet(pretrained=False)
    runner = Runner("googlenet", net, shape, opset_version)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_mobilenet_v2(shape = [1, 3, 224, 224], opset_version=9):
    net = models.mobilenet_v2(pretrained=False)
    runner = Runner("mobilenet", net, shape, opset_version)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_mobilenet_v3(shape = [1, 3, 224, 224], opset_version=13):
    '''
    symbolic_opset13.py
    @parse_args("v")
    def hardswish(g, self):
        return g.op("HardSwish", self)
    '''
    net = models.mobilenet_v3_small(pretrained=False)
    runner = Runner("mobilenet_v3", net, shape, opset_version)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_densenet121(shape = [1, 3, 224, 224], opset_version=9):
    net = models.densenet121(pretrained=False)
    runner = Runner("densenet121", net, shape, opset_version)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_densenet161(shape = [1, 3, 224, 224], opset_version=9):
    net = models.densenet161(pretrained=False)
    runner = Runner("densenet161", net, shape, opset_version)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()   

def test_inception_v3(shape = [1, 3, 299, 299], opset_version=9):
    net = models.inception_v3(pretrained=False)
    runner = Runner("inception_v3", net, shape, opset_version)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()  

def test_vgg16(shape = [1, 3, 224, 224], opset_version=9):
    net = models.vgg16(pretrained=False)
    runner = Runner("vgg16", net, shape, opset_version)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_ssd300_vgg16(shape = [1, 3, 300, 300], opset_version=13):
    '''
    symbolic_opset13.py
    @parse_args('v', 'v', 'v', 'i', 'i', 'i')
    def linalg_norm(g, self):
        return g.op("LpNormalization", self)
    '''    
    from models.ssd import build_ssd
    net = build_ssd("export")
    # import torch
    # state_dict = torch.load("test/ssd_300_VOC.pth", map_location=torch.device("cpu"))
    # state_dict["L2Norm.weight"] = state_dict["L2Norm.weight"].unsqueeze(0).unsqueeze(2).unsqueeze(3)
    # net.load_state_dict(state_dict)

    runner = Runner("ssd300_vgg16", net, shape, opset_version)
    runner.pyotrch_inference()
    net_ = build_ssd("export", export_mode=True)
    runner.convert(net_)
    runner.caffe_inference()
    runner.check_result()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pytest.main(['-p', 'no:warnings', '-v', 'test'])
