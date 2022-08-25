import os
import sys
import torch
import pytest
import warnings

from bin.fx.utils import OnnxBaseTester as Tester

def test_Linear_basic(shape = [1, 3], opset_version=13):
    model = torch.nn.Linear(3, 5, bias=True)
    Tester("Linear_basic", model, shape, opset_version)

def test_Linear_nobias(shape = [1, 3], opset_version=13):
    model = torch.nn.Linear(3, 5, bias=False)
    Tester("Linear_nobias", model, shape, opset_version)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pytest.main(['-p', 'no:warnings', '-v', 'test/op_test/caffe/linear/test_linear.py'])
