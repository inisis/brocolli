import torch
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
        model = torch.nn.Linear(3, 5, bias=bias)
        x = torch.rand(shape)
        Tester(request.node.name, model, x)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(
        ["-p", "no:warnings", "-v", "test/converter/op_test/onnx/linear/test_linear.py"]
    )
