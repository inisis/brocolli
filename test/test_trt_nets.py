import os
os.environ[
    "GLOG_minloglevel"
] = "3"  # 0 - debug 1 - info (still a LOT of outputs) 2 - warnings 3 - errors

import pytest
import warnings

import torchvision.models as models

from bin.pytorch2caffe import Runner

FUSE = True

os.makedirs('tmp', exist_ok=True)

def test_resnet18(shape = [1, 3, 224, 224], opset_version=9):
    net = models.resnet18(pretrained=False)
    runner = Runner("resnet18", net, shape, opset_version)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pytest.main(['-p', 'no:warnings', '-v', 'test/test_trt_nets.py'])
