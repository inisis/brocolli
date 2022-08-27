import os
import sys
import torch
import pytest
import warnings

from bin.converter.utils import CaffeBaseTester as Tester

def test_Conv2d_basic(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=1, stride=1, padding=0)
    Tester("Conv2d_basic", model, shape, opset_version)

def test_Conv2d_kernel_3x3(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=(3, 3), stride=1, padding=0)
    Tester("Conv2d_kernel_3x3", model, shape, opset_version)

def test_Conv2d_kernel_1x3(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=(1, 3), stride=1, padding=0)
    Tester("Conv2d_kernel_1x3", model, shape, opset_version)

def test_Conv2d_kernel_3x1(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=(3, 1), stride=1, padding=0)
    Tester("Conv2d_kernel_3x1", model, shape, opset_version)

def test_Conv2d_stride_3x3(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=1, stride=(3, 3), padding=0)
    Tester("Conv2d_stride_3x3", model, shape, opset_version)

def test_Conv2d_stride_3x1(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=1, stride=(3, 1), padding=0)
    Tester("Conv2d_stride_3x1", model, shape, opset_version)

def test_Conv2d_stride_1x3(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=1, stride=(1, 3), padding=0)
    Tester("Conv2d_stride_1x3", model, shape, opset_version)

def test_Conv2d_padding_3x3(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=1, stride=1, padding=(3, 3))
    Tester("Conv2d_padding_3x3", model, shape, opset_version)

def test_Conv2d_padding_3x1(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=1, stride=1, padding=(3, 1))
    Tester("Conv2d_padding_3x1", model, shape, opset_version)

def test_Conv2d_padding_1x3(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=1, stride=1, padding=(1, 3))
    Tester("Conv2d_padding_1x3", model, shape, opset_version)

def test_Conv2d_dilation_3x3(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=1, stride=1, padding=0, dilation=(3, 3))
    Tester("Conv2d_dilation_3x3", model, shape, opset_version)

def test_Conv2d_dilation_3x1(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=1, stride=1, padding=0, dilation=(3, 1))
    Tester("Conv2d_dilation_3x1", model, shape, opset_version)

def test_Conv2d_dilation_1x3(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=1, stride=1, padding=0, dilation=(1, 3))
    Tester("Conv2d_dilation_1x3", model, shape, opset_version)    


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pytest.main(['-p', 'no:warnings', '-v', 'test/op_test/caffe/conv/test_conv.py'])
