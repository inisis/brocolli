import torch
import torch.nn.functional as F
import pytest
import warnings

from brocolli.testing.common_utils import OnnxBaseTester as Tester


class TestLinearClass:
    @pytest.mark.parametrize("bias", (True, False))
    def test_Linear(
        self,
        request,
        bias,
        shape=(1, 3),
    ):
        class Linear(torch.nn.Module):
            def __init__(self, bias):
                super(Linear, self).__init__()
                self.weight = torch.nn.Parameter(torch.FloatTensor(4, 3))
                self.bias = None
                if bias:
                    self.bias = torch.nn.Parameter(torch.FloatTensor(4))

            def forward(self, x):
                return F.linear(x, self.weight, self.bias)

        model = Linear(bias=bias)
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    @pytest.mark.parametrize("bias", (True, False))
    def test_Linear_Module(
        self,
        request,
        bias,
        shape=(1, 3),
    ):
        class Linear(torch.nn.Module):
            def __init__(self, bias):
                super(Linear, self).__init__()
                self.linear = torch.nn.Linear(3, 5, bias)

            def forward(self, x):
                return self.linear(x)

        model = Linear(bias=bias)
        x = torch.rand(shape)
        Tester(request.node.name, model, x)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(
        ["-p", "no:warnings", "-v", "test/converter/op_test/onnx/linear/test_linear.py"]
    )
