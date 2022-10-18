import os
import sys
import torch
import pytest
import warnings

from brocolli.testing.common_utils import CaffeBaseTester as Tester


class TestArithmeticClass:
    def test_Add(
        self,
        request,
        shape=([1, 3, 1, 1], [1, 3, 1, 1]),
    ):
        class Add(torch.nn.Module):
            def __init__(self):
                super(Add, self).__init__()

            def forward(self, x, y):
                return x + y

        model = Add()
        Tester(request.node.name, model, shape)

    def test_TorchAdd(
        self,
        request,
        shape=([1, 3, 1, 1], [1, 3, 1, 1]),
    ):
        class TorchAdd(torch.nn.Module):
            def __init__(self):
                super(TorchAdd, self).__init__()

            def forward(self, x, y):
                return torch.add(x, y)

        model = TorchAdd()
        Tester(request.node.name, model, shape)

    def test_Mean_keepdim(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        class Mean(torch.nn.Module):
            def __init__(self, dim, keepdim):
                super(Mean, self).__init__()
                self.dim = dim
                self.keepdim = keepdim

            def forward(self, x):
                return x.mean(self.dim, self.keepdim)

        model = Mean((2, 3), True)
        Tester(request.node.name, model, shape)

    def test_Mul(
        self,
        request,
        shape=([1, 3, 1, 1], [1, 3, 1, 1]),
    ):
        class Mul(torch.nn.Module):
            def __init__(self):
                super(Mul, self).__init__()

            def forward(self, x, y):
                return x * y

        model = Mul()
        Tester(request.node.name, model, shape)

    def test_TorchMul(
        self,
        request,
        shape=([1, 3, 1, 1], [1, 3, 1, 1]),
    ):
        class TorchMul(torch.nn.Module):
            def __init__(self):
                super(TorchMul, self).__init__()

            def forward(self, x, y):
                return torch.mul(x, y)

        model = TorchMul()
        Tester(request.node.name, model, shape)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(
        ["-p", "no:warnings", "-v", "test/op_test/caffe/arithmetic/test_arithmetic.py"]
    )
