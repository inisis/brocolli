import os
os.environ[
    "GLOG_minloglevel"
] = "3"  # 0 - debug 1 - info (still a LOT of outputs) 2 - warnings 3 - errors

import pytest
import warnings

import torchvision.models as models

from bin.fx.pytorch2caffe import Runner

FUSE = True

os.makedirs('tmp', exist_ok=True)

def test_alexnet(shape = [1, 3, 224, 224], opset_version=9, fuse=FUSE):
    net = models.alexnet(pretrained=False)
    runner = Runner("alexnet", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_resnet18(shape = [1, 3, 224, 224], opset_version=9, fuse=FUSE):
    net = models.resnet18(pretrained=False)
    runner = Runner("resnet18", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_squeezenet(shape = [1, 3, 227, 227], opset_version=9, fuse=FUSE):
    net = models.squeezenet1_0(pretrained=False)
    runner = Runner("squeezenet", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_googlenet(shape = [1, 3, 224, 224], opset_version=13, fuse=FUSE):
    net = models.googlenet(pretrained=False)
    runner = Runner("googlenet", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_mobilenet_v2(shape = [1, 3, 224, 224], opset_version=9, fuse=FUSE):
    net = models.mobilenet_v2(pretrained=False)
    runner = Runner("mobilenet", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_mobilenet_v3(shape = [1, 3, 224, 224], opset_version=13, fuse=FUSE):
    net = models.mobilenet_v3_small(pretrained=False)
    runner = Runner("mobilenet_v3", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_densenet121(shape = [1, 3, 224, 224], opset_version=9, fuse=FUSE):
    net = models.densenet121(pretrained=False)
    runner = Runner("densenet121", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_densenet161(shape = [1, 3, 224, 224], opset_version=9, fuse=FUSE):
    net = models.densenet161(pretrained=False)
    runner = Runner("densenet161", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_shufflenet(shape = [1, 3, 224, 224], opset_version=9, fuse=FUSE):
    net = models.shufflenet_v2_x1_0(pretrained=False)
    runner = Runner("shufflenet_v2_x1_0", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pytest.main(['-p', 'no:warnings', '-v', 'test/fx/test_caffe_nets.py'])
