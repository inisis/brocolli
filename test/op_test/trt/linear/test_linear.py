import os
import sys
import torch
import pytest
import warnings

from bin.utils import TensorRTBaseTester as Tester

def test_Linear_basic(shape = [1, 3], opset_version=13):
    model = torch.nn.Linear(3, 5)
    Tester("Linear_basic", model, shape, opset_version)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pytest.main(['-p', 'no:warnings', '-v', 'test/op_test/trt/linear/test_linear.py'])
