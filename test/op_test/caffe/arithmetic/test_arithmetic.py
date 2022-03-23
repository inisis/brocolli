import os
import sys
import torch
import pytest
import warnings

from bin.utils import CaffeBaseTester as Tester

def test_Add(shape = ([1, 3, 1, 1], [1, 3, 1, 1]), opset_version=13):
    class Add(torch.nn.Module):
        def __init__(self):
            super(Add, self).__init__()

        def forward(self, x, y):
            return x + y    
    
    model = Add()
    Tester("Add", model, shape, opset_version)

def test_IAdd(shape = ([1, 3, 1, 1], [1, 3, 1, 1]), opset_version=13):
    class IAdd(torch.nn.Module):
        def __init__(self):
            super(IAdd, self).__init__()

        def forward(self, x, y):
            x += y
            return x 
    
    model = IAdd()
    Tester("IAdd", model, shape, opset_version)

def test_TorchAdd(shape = ([1, 3, 1, 1], [1, 3, 1, 1]), opset_version=13):
    class TorchAdd(torch.nn.Module):
        def __init__(self):
            super(TorchAdd, self).__init__()

        def forward(self, x, y):
            return torch.add(x, y)

    model = TorchAdd()
    Tester("TorchAdd", model, shape, opset_version)

class Mean(torch.nn.Module):
    def __init__(self, dim, keepdim):
        super(Mean, self).__init__()
        self.dim = dim
        self.keepdim = keepdim
    def forward(self, x):
        return x.mean(self.dim, self.keepdim)

def test_Mean_keepdim(shape = [1, 3, 32, 32], opset_version=13):
    model = Mean((2, 3), True)
    Tester("Mean_keepdim", model, shape, opset_version)

def test_Mul(shape = ([1, 3, 1, 1], [1, 3, 1, 1]), opset_version=13):
    class Mul(torch.nn.Module):
        def __init__(self):
            super(Mul, self).__init__()

        def forward(self, x, y):
            return x * y
    
    model = Mul()
    Tester("Mul", model, shape, opset_version)

def test_IMul(shape = ([1, 3, 1, 1], [1, 3, 1, 1]), opset_version=13):
    class IMul(torch.nn.Module):
        def __init__(self):
            super(IMul, self).__init__()

        def forward(self, x, y):
            x *= y
            return x 
    
    model = IMul()
    Tester("IMul", model, shape, opset_version)

def test_TorchMul(shape = ([1, 3, 1, 1], [1, 3, 1, 1]), opset_version=13):
    class TorchMul(torch.nn.Module):
        def __init__(self):
            super(TorchMul, self).__init__()

        def forward(self, x, y):
            return torch.mul(x, y)

    model = TorchMul()
    Tester("TorchMul", model, shape, opset_version)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pytest.main(['-p', 'no:warnings', '-v', 'test/op_test/caffe/arithmetic/test_arithmetic.py'])
