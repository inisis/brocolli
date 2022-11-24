import torch
import pytest
import warnings

from brocolli.testing.common_utils import OnnxBaseTester as Tester


def test_Add(
    shape=([1, 3, 1, 1], [1, 3, 1, 1]),
):
    class Add(torch.nn.Module):
        def __init__(self):
            super(Add, self).__init__()

        def forward(self, x, y):
            print("x", x.shape)
            print("y", y.shape)
            z = x + y
            print("z", z)
            return x + y

    model = Add()
    Tester("Add", model, shape)


def test_TorchAdd(
    shape=([1, 3, 1, 1], [1, 3, 1, 1]),
):
    class TorchAdd(torch.nn.Module):
        def __init__(self):
            super(TorchAdd, self).__init__()

        def forward(self, x, y):
            return torch.add(x, y)

    model = TorchAdd()
    Tester("TorchAdd", model, shape)


class Mean(torch.nn.Module):
    def __init__(self, dim, keepdim):
        super(Mean, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return x.mean(self.dim, self.keepdim)


def test_Mean_keepdim(
    shape=[1, 3, 32, 32],
):
    model = Mean((2, 3), True)
    Tester("Mean_keepdim", model, shape)


def test_Mul(
    shape=([1, 3, 1, 1], [1, 3, 1, 1]),
):
    class Mul(torch.nn.Module):
        def __init__(self):
            super(Mul, self).__init__()

        def forward(self, x, y):
            return x * y

    model = Mul()
    Tester("Mul", model, shape)


def test_TorchMul(
    shape=([1, 3, 1, 1], [1, 3, 1, 1]),
):
    class TorchMul(torch.nn.Module):
        def __init__(self):
            super(TorchMul, self).__init__()

        def forward(self, x, y):
            return torch.mul(x, y)

    model = TorchMul()
    Tester("TorchMul", model, shape)


def test_Abs(
    shape=[1, 3, 1, 1],
):
    class Abs(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.abs(x)

    model = Abs()
    Tester("Abs", model, shape)


def test_Abs_1(
    shape=[1, 3, 1, 1],
):
    class Abs(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x.abs()

    model = Abs()
    Tester("Abs_1", model, shape)


def test_Cos(
    shape=[1, 3, 1, 1],
):
    class Cos(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.cos(x)

    model = Cos()
    Tester("Cos", model, shape)


def test_Cos_1(
    shape=[1, 3, 1, 1],
):
    class Cos(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x.cos()

    model = Cos()
    Tester("Cos_1", model, shape)


def test_Sin(
    shape=[1, 3, 1, 1],
):
    class Sin(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sin(x)

    model = Sin()
    Tester("Sin", model, shape)


def test_Sin_1(
    shape=[1, 3, 1, 1],
):
    class Sin(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x.sin()

    model = Sin()
    Tester("Sin_1", model, shape)


def test_Pow(
    shape=[1, 3, 1, 1],
):
    class Pow(torch.nn.Module):
        def __init__(self, exp=2):
            super(Pow, self).__init__()
            self.exp = exp

        def forward(self, x):
            return torch.pow(x, self.exp)

    model = Pow()
    Tester("Pow", model, shape)


def test_Pow_1(
    shape=[1, 3, 1, 1],
):
    class Pow(torch.nn.Module):
        def __init__(self, exp=2):
            super(Pow, self).__init__()
            self.exp = exp

        def forward(self, x):
            return x.pow(self.exp)

    model = Pow()
    Tester("Pow_1", model, shape)


def test_Log(
    shape=[1, 3, 1, 1],
):
    class Log(torch.nn.Module):
        def __init__(self):
            super(Log, self).__init__()

        def forward(self, x):
            return torch.log(x)

    model = Log()
    Tester("Log", model, shape)


def test_Log_1(
    shape=[1, 3, 1, 1],
):
    class Log(torch.nn.Module):
        def __init__(self):
            super(Log, self).__init__()

        def forward(self, x):
            return x.log()

    model = Log()
    Tester("Log_1", model, shape)


def test_Sqrt(
    shape=[1, 3, 1, 1],
):
    class Sqrt(torch.nn.Module):
        def __init__(self):
            super(Sqrt, self).__init__()

        def forward(self, x):
            return torch.sqrt(x)

    model = Sqrt()
    Tester("Sqrt", model, shape)


def test_Sqrt_1(
    shape=[1, 3, 1, 1],
):
    class Sqrt(torch.nn.Module):
        def __init__(self):
            super(Sqrt, self).__init__()

        def forward(self, x):
            return x.sqrt()

    model = Sqrt()
    Tester("Sqrt_1", model, shape)


def test_ReduceSum(
    shape=[1, 3, 1, 1],
):
    class ReduceSum(torch.nn.Module):
        def __init__(self, dim, keep_dim=False):
            super().__init__()
            self.dim = dim
            self.keep_dim = keep_dim

        def forward(self, x):
            return torch.sum(x, self.dim, self.keep_dim)

    model = ReduceSum(dim=1)
    Tester("ReduceSum", model, shape)


def test_ReduceMean(
    shape=[1, 3, 1, 1],
):
    class ReduceMean(torch.nn.Module):
        def __init__(self, dim, keep_dim=False):
            super().__init__()
            self.dim = dim
            self.keep_dim = keep_dim

        def forward(self, x):
            return torch.mean(x, self.dim, self.keep_dim)

    model = ReduceMean(dim=1)
    Tester("ReduceMean", model, shape)


def test_Sub(
    shape=([1, 3, 1, 1], [1, 3, 1, 1]),
):
    class Sub(torch.nn.Module):
        def __init__(self):
            super(Sub, self).__init__()

        def forward(self, x, y):
            return torch.sub(x, y)

    model = Sub()
    Tester("Sub", model, shape)


def test_Min(
    shape=([1, 3, 1, 1], [1, 3, 1, 1]),
):
    class Min(torch.nn.Module):
        def __init__(self):
            super(Min, self).__init__()

        def forward(self, x, y):
            return torch.min(x, y)

    model = Min()
    Tester("Min", model, shape)


# def test_Max(shape=([1, 3, 1, 1], [1, 3, 1, 1]),):
#     class Max(torch.nn.Module):
#         def __init__(self):
#             super(Max, self).__init__()

#         def forward(self, x, y):
#             return torch.max(x, y)

#     model = Max()
#     Tester("Max", model, shape)


def test_Div(
    shape=([1, 3, 1, 1], [1, 3, 1, 1]),
):
    class Div(torch.nn.Module):
        def __init__(self):
            super(Div, self).__init__()

        def forward(self, x, y):
            return torch.div(x, y)

    model = Div()
    Tester("Div", model, shape)


def test_Matmul(
    shape=([1, 3, 1, 1], [1, 3, 1, 1]),
):
    class Matmul(torch.nn.Module):
        def __init__(self):
            super(Matmul, self).__init__()

        def forward(self, x, y):
            return torch.matmul(x, y)

    model = Matmul()
    Tester("Matmul", model, shape)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(
        ["-p", "no:warnings", "-v", "test/op_test/onnx/arithmetic/test_arithmetic.py"]
    )
