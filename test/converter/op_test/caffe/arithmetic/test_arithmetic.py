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
        x = torch.rand(shape[0])
        y = torch.rand(shape[1])
        Tester(request.node.name, model, (x, y))

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
        x = torch.rand(shape[0])
        y = torch.rand(shape[1])
        Tester(request.node.name, model, (x, y))

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
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

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
        x = torch.rand(shape[0])
        y = torch.rand(shape[1])
        Tester(request.node.name, model, (x, y))

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
        x = torch.rand(shape[0])
        y = torch.rand(shape[1])
        Tester(request.node.name, model, (x, y))

    def test_power(
        self,
        request,
        shape=(1, 3, 1, 1),
    ):
        class Power(torch.nn.Module):
            def __init__(self):
                super(Power, self).__init__()

            def forward(self, x):
                return x.pow(2)

        model = Power()
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    def test_TorchPower(
        self,
        request,
        shape=(1, 3, 1, 1),
    ):
        class TorchPower(torch.nn.Module):
            def __init__(self):
                super(TorchPower, self).__init__()

            def forward(self, x):
                return torch.pow(x, 2)

        model = TorchPower()
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    def test_Power(
        self,
        request,
        shape=(1, 3, 1, 1),
    ):
        class TorchPower(torch.nn.Module):
            def __init__(self):
                super(TorchPower, self).__init__()

            def forward(self, x):
                return x ** 2

        model = TorchPower()
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(
        [
            "-p",
            "no:warnings",
            "-v",
            "test/converter/op_test/caffe/arithmetic/test_arithmetic.py",
        ]
    )
