import os
import sys
import torch
import pytest
import warnings

from brocolli.testing.common_utils import CaffeBaseTester as Tester


class TestLinearClass:
    def test_Linear_basic(
        self,
        request,
        shape=[1, 3],
    ):
        model = torch.nn.Linear(3, 5, bias=True)
        x = torch.rand(shape)
        Tester(request.node.name, model, x)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(
        [
            "-p",
            "no:warnings",
            "-v",
            "test/converter/op_test/caffe/linear/test_linear.py",
        ]
    )
