import os
import sys
import torch
import pytest
import warnings

from bin.utils import CaffeBaseTester as Tester

def test_LeakyRelu(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.LeakyReLU()
    Tester("LeakyRelu", model, shape, opset_version)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pytest.main(['-p', 'no:warnings', '-v', 'test/op_test/caffe/activation/test_activation.py'])
