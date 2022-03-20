import os
import sys
import torch
import pytest
import warnings

from bin.utils import CaffeBaseTester as Tester

def test_pooling(shape = [1, 1, 32, 32], opset_version=13):
    model = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
    tester = Tester("pooling", model, shape, opset_version)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pytest.main(['-p', 'no:warnings', '-v', 'test/op_test/caffe/pooling/test_pooling.py'])
