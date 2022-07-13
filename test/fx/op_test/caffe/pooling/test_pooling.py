import os
import sys
import torch
import pytest
import warnings

from bin.fx.utils import CaffeBaseTester as Tester

def test_AdaptiveAvgPool2d_1x1(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.AdaptiveAvgPool2d((1, 1))
    Tester("AdaptiveAvgPool2d_1x1", model, shape, opset_version)

def test_AdaptiveAvgPool2d_1x2(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.AdaptiveAvgPool2d((1, 2))
    Tester("AdaptiveAvgPool2d_1x2", model, shape, opset_version)

def test_AdaptiveAvgPool2d_2x1(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.AdaptiveAvgPool2d((2, 1))
    Tester("AdaptiveAvgPool2d_2x1", model, shape, opset_version)  

def test_AvgPool2d_without_ceil_mode(shape = [1, 1, 32, 32], opset_version=13):
    model = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
    Tester("AvgPool2d_without_ceil_mode", model, shape, opset_version)    

def test_AvgPool2d_with_ceil_mode(shape = [1, 1, 32, 32], opset_version=13):
    model = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
    Tester("AvgPool2d_with_ceil_mode", model, shape, opset_version)   

def test_MaxPool2d_with_return_indices(shape = [1, 1, 32, 32], opset_version=13):
    model = torch.nn.MaxPool2d(2, 2, return_indices=True)
    Tester("MaxPool2d_with_return_indices", model, shape, opset_version)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pytest.main(['-p', 'no:warnings', '-v', 'test/op_test/caffe/pooling/test_pooling.py'])
