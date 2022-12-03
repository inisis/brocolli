import torch
import pytest
import warnings

from brocolli.testing.common_utils import OnnxBaseTester as Tester


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

        x = torch.rand(shape[0])
        y = torch.rand(shape[1])
        model = Add()
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

        x = torch.rand(shape[0])
        y = torch.rand(shape[1])
        model = TorchAdd()
        Tester(request.node.name, model, (x, y))

    @pytest.mark.parametrize("keepdim", (True, False))
    def test_Mean_keepdim(
        self,
        request,
        keepdim,
        shape=[1, 3, 32, 32],
    ):
        class Mean(torch.nn.Module):
            def __init__(self, dim, keepdim):
                super(Mean, self).__init__()
                self.dim = dim
                self.keepdim = keepdim

            def forward(self, x):
                return x.mean(self.dim, self.keepdim)

        x = torch.rand(shape)
        model = Mean((2, 3), keepdim)
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

        x = torch.rand(shape[0])
        y = torch.rand(shape[1])
        model = Mul()
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

        x = torch.rand(shape[0])
        y = torch.rand(shape[1])
        model = TorchMul()
        Tester(request.node.name, model, (x, y))

    def test_Abs(
        self,
        request,
        shape=[1, 3, 1, 1],
    ):
        class Abs(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.abs(x) + x.abs()

        x = torch.rand(shape)
        model = Abs()
        Tester(request.node.name, model, x)

    def test_Cos(
        self,
        request,
        shape=[1, 3, 1, 1],
    ):
        class Cos(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.cos(x) + x.cos()

        x = torch.rand(shape)
        model = Cos()
        Tester(request.node.name, model, x)

    def test_Sin(
        self,
        request,
        shape=[1, 3, 1, 1],
    ):
        class Sin(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.sin(x) + x.sin()

        x = torch.rand(shape)
        model = Sin()
        Tester(request.node.name, model, x)

    def test_Pow(
        self,
        request,
        shape=[1, 3, 1, 1],
    ):
        class Pow(torch.nn.Module):
            def __init__(self, exp=2):
                super(Pow, self).__init__()
                self.exp = exp

            def forward(self, x):
                return torch.pow(x, self.exp) + x.pow(self.exp)

        x = torch.rand(shape)
        model = Pow()
        Tester(request.node.name, model, x)

    def test_Log(
        self,
        request,
        shape=[1, 3, 1, 1],
    ):
        class Log(torch.nn.Module):
            def __init__(self):
                super(Log, self).__init__()

            def forward(self, x):
                return torch.log(x) + x.log()

        x = torch.rand(shape)
        model = Log()
        Tester(request.node.name, model, x)

    def test_Sqrt(
        self,
        request,
        shape=[1, 3, 1, 1],
    ):
        class Sqrt(torch.nn.Module):
            def __init__(self):
                super(Sqrt, self).__init__()

            def forward(self, x):
                return torch.sqrt(x) + x.sqrt()

        x = torch.rand(shape)
        model = Sqrt()
        Tester(request.node.name, model, x)

    def test_ReduceSum(
        self,
        request,
        shape=[1, 3, 1, 1],
    ):
        class ReduceSum(torch.nn.Module):
            def __init__(self, dim, keep_dim=False):
                super().__init__()
                self.dim = dim
                self.keep_dim = keep_dim

            def forward(self, x):
                return torch.sum(x, self.dim, self.keep_dim)

        x = torch.rand(shape)
        model = ReduceSum(dim=1)
        Tester(request.node.name, model, x)

    def test_ReduceMean(
        self,
        request,
        shape=[1, 3, 1, 1],
    ):
        class ReduceMean(torch.nn.Module):
            def __init__(self, dim, keep_dim=False):
                super().__init__()
                self.dim = dim
                self.keep_dim = keep_dim

            def forward(self, x):
                return torch.mean(x, self.dim, self.keep_dim)

        x = torch.rand(shape)
        model = ReduceMean(dim=1)
        Tester(request.node.name, model, x)

    def test_Sub(
        self,
        request,
        shape=([1, 3, 1, 1], [1, 3, 1, 1]),
    ):
        class Sub(torch.nn.Module):
            def __init__(self):
                super(Sub, self).__init__()

            def forward(self, x, y):
                return torch.sub(x, y)

        x = torch.rand(shape[0])
        y = torch.rand(shape[1])
        model = Sub()
        Tester(request.node.name, model, (x, y))

    def test_Min(
        self,
        request,
        shape=([1, 3, 1, 1], [1, 3, 1, 1]),
    ):
        class Min(torch.nn.Module):
            def __init__(self):
                super(Min, self).__init__()

            def forward(self, x, y):
                return torch.min(x, y)

        x = torch.rand(shape[0])
        y = torch.rand(shape[1])
        model = Min()
        Tester(request.node.name, model, (x, y))

    def test_Div(
        self,
        request,
        shape=([1, 3, 1, 1], [1, 3, 1, 1]),
    ):
        class Div(torch.nn.Module):
            def __init__(self):
                super(Div, self).__init__()

            def forward(self, x, y):
                return torch.div(x, y)

        x = torch.rand(shape[0])
        y = torch.rand(shape[1])
        model = Div()
        Tester(request.node.name, model, (x, y))

    def test_Matmul(
        self,
        request,
        shape=([1, 3, 1, 1], [1, 3, 1, 1]),
    ):
        class Matmul(torch.nn.Module):
            def __init__(self):
                super(Matmul, self).__init__()

            def forward(self, x, y):
                return torch.matmul(x, y)

        x = torch.rand(shape[0])
        y = torch.rand(shape[1])
        model = Matmul()
        Tester(request.node.name, model, (x, y))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(
        [
            "-p",
            "no:warnings",
            "-v",
            "test/converter/op_test/onnx/arithmetic/test_arithmetic.py",
        ]
    )
