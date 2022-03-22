import os
import sys
import torch
import pytest
import warnings

from bin.utils import CaffeBaseTester as Tester

def test_Conv2d_basic(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=1, stride=1, padding=0)
    Tester("Conv2d_basic", model, shape, opset_version)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pytest.main(['-p', 'no:warnings', '-v', 'test/op_test/caffe/activation/test_activation.py'])
