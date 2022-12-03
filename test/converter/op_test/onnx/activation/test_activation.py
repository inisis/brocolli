import torch
import torch.nn.functional as F

import pytest
import warnings

from brocolli.testing.common_utils import OnnxBaseTester as Tester


class TestActivationClass:
    def test_ReLU(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        x = torch.rand(shape)
        model = torch.nn.ReLU()
        Tester(request.node.name, model, x)

    def test_PReLU(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        x = torch.rand(shape)
        model = torch.nn.PReLU()
        Tester(request.node.name, model, x)

    def test_Sigmoid(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        x = torch.rand(shape)
        model = torch.nn.Sigmoid()
        Tester(request.node.name, model, x)

    @pytest.mark.parametrize("dim", (None, -1, 1))
    def test_Softmax_basic(self, request, dim, shape=[1, 3, 32, 32]):
        x = torch.rand(shape)
        model = torch.nn.Softmax(dim)
        Tester(request.node.name, model, x)

    def test_Softmax_module(self, request, shape=[1, 3, 32, 32]):
        class Softmax(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.softmax = torch.nn.Softmax()

            def forward(self, x):
                return self.softmax(x)

        x = torch.rand(shape)
        model = Softmax()
        Tester(request.node.name, model, x)

    def test_Hardsigmoid(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        x = torch.rand(shape)
        model = torch.nn.Hardsigmoid()
        Tester(request.node.name, model, x)

    def test_Hardswish(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        x = torch.rand(shape)
        model = torch.nn.Hardswish()
        Tester(request.node.name, model, x)

    def test_CReLU(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        class CReLU(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.cat((F.relu(x), F.relu(-x)), 1)

        x = torch.rand(shape)
        model = CReLU()
        Tester(request.node.name, model, x)

    @pytest.mark.parametrize("negative_slope", (0.01, 0.1))
    def test_LeakyReLU(self, request, negative_slope, shape=[1, 3, 32, 32]):
        class LeakyReLU(torch.nn.Module):
            def __init__(self, negative_slope=0.01):
                super().__init__()
                self.lrelu = torch.nn.LeakyReLU(negative_slope)

            def forward(self, x):
                out = self.lrelu(x)

                return out

        x = torch.rand(shape)
        model = LeakyReLU(negative_slope)

        Tester(request.node.name, model, x)

    def test_Softplus(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        x = torch.rand(shape)
        model = torch.nn.Softplus()
        Tester(request.node.name, model, x)

    def test_SELU(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        x = torch.rand(shape)
        model = torch.nn.SELU()
        Tester(request.node.name, model, x)

    def test_SELU_module(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        class SELU(torch.nn.Module):
            def __init__(self, negative_slope=0.01):
                super().__init__()
                self.selu = torch.nn.SELU()

            def forward(self, x):
                return self.selu(x)

        x = torch.rand(shape)
        model = SELU()
        Tester(request.node.name, model, x)

    def test_ELU(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        x = torch.rand(shape)
        model = torch.nn.ELU()
        Tester(request.node.name, model, x)

    def test_ELU_module(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        class ELU(torch.nn.Module):
            def __init__(self, negative_slope=0.01):
                super().__init__()
                self.elu = torch.nn.ELU()

            def forward(self, x):
                return self.elu(x)

        x = torch.rand(shape)
        model = ELU()
        Tester(request.node.name, model, x)

    def test_Softplus(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        x = torch.rand(shape)
        model = torch.nn.Softplus()
        Tester(request.node.name, model, x)

    def test_Softplus_module(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        class Softplus(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.softplus = torch.nn.Softplus()

            def forward(self, x):
                return self.softplus(x)

        x = torch.rand(shape)
        model = Softplus()
        Tester(request.node.name, model, x)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(
        [
            "-p",
            "no:warnings",
            "-v",
            "test/converter/op_test/onnx/activation/test_activation.py",
        ]
    )
