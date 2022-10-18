import torch
import torch.nn.functional as F

import pytest
import warnings

from brocolli.testing.common_utils import OnnxBaseTester as Tester


def test_ReLU(
    shape=[1, 3, 32, 32],
):
    model = torch.nn.ReLU()
    Tester("ReLU", model, shape)


def test_PReLU(
    shape=[1, 3, 32, 32],
):
    model = torch.nn.PReLU(num_parameters=3)
    Tester("PReLU", model, shape)


def test_LeakyReLU(
    shape=[1, 3, 32, 32],
):
    model = torch.nn.LeakyReLU()
    Tester("LeakyReLU", model, shape)


def test_ReLU6(
    shape=[1, 3, 32, 32],
):
    model = torch.nn.ReLU6()
    Tester("ReLU6", model, shape)


def test_Sigmoid(
    shape=[1, 3, 32, 32],
):
    model = torch.nn.Sigmoid()
    Tester("Sigmoid", model, shape)


class Sigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)


def test_Sigmoid_module(
    shape=[1, 3, 32, 32],
):
    model = Sigmoid()
    Tester("Sigmoid_module", model, shape)


def test_Softmax_basic(
    shape=[1, 3, 32, 32],
):
    model = torch.nn.Softmax()
    Tester("Softmax_basic", model, shape)


def test_Softmax_dim_2(
    shape=[1, 3, 32, 32],
):
    model = torch.nn.Softmax(dim=2)
    Tester("Softmax_dim_2", model, shape)


def test_Softmax_dim_3(
    shape=[1, 3, 32, 32],
):
    model = torch.nn.Softmax(dim=2)
    Tester("Softmax_dim_3", model, shape)


class Softmax(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        return self.softmax(x)


def test_Softmax_module(
    shape=[1, 3, 32, 32],
):
    model = Softmax()
    Tester("Softmax_module", model, shape)


def test_Hardsigmoid(
    shape=[1, 3, 32, 32],
):
    model = torch.nn.Hardsigmoid()
    Tester("Hardsigmoid", model, shape)


def test_Hardswish(
    shape=[1, 3, 32, 32],
):
    model = torch.nn.Hardswish()
    Tester("Hardswish", model, shape)


def test_CReLU(
    shape=[1, 3, 32, 32],
):
    class CReLU(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.cat((F.relu(x), F.relu(-x)), 1)

    model = CReLU()
    Tester("CReLU", model, shape)


class LeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.lrelu = torch.nn.LeakyReLU(negative_slope)

    def forward(self, x):
        return self.lrelu(x)


def test_LeakyReLU(
    shape=[1, 3, 32, 32],
):
    model = LeakyReLU()
    Tester("LeakyReLU", model, shape)


def test_LeakyReLU_1(
    shape=[1, 3, 32, 32],
):
    model = LeakyReLU(0.1)
    Tester("LeakyReLU_1", model, shape)


def test_Softplus(
    shape=[1, 3, 32, 32],
):
    model = torch.nn.Softplus()
    Tester("Softplus", model, shape)


def test_SELU(
    shape=[1, 3, 32, 32],
):
    model = torch.nn.SELU()
    Tester("SELU", model, shape)


class SELU(torch.nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.selu = torch.nn.SELU()

    def forward(self, x):
        return self.selu(x)


def test_SELU_module(
    shape=[1, 3, 32, 32],
):
    model = SELU()
    Tester("SELU_module", model, shape)


def test_ELU(
    shape=[1, 3, 32, 32],
):
    model = torch.nn.ELU()
    Tester("ELU", model, shape)


class ELU(torch.nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.elu = torch.nn.ELU()

    def forward(self, x):
        return self.elu(x)


def test_ELU_module(
    shape=[1, 3, 32, 32],
):
    model = ELU()
    Tester("ELU_module", model, shape)


def test_Softplus(
    shape=[1, 3, 32, 32],
):
    model = torch.nn.Softplus()
    Tester("Softplus", model, shape)


class Softplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        return self.softplus(x)


def test_Softplus_module(
    shape=[1, 3, 32, 32],
):
    model = Softplus()
    Tester("Softplus_module", model, shape)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(
        ["-p", "no:warnings", "-v", "test/op_test/caffe/activation/test_activation.py"]
    )
