import os
import sys
import torch
import pytest
import warnings

from brocolli.testing.common_utils import CaffeBaseTester as Tester


class TestLinearClass:
    def test_BatchNorm1d(
        self,
        request,
        shape=[1, 3, 32],
    ):
        class Normalization(torch.nn.Module):
            def __init__(self):
                super(Normalization, self).__init__()
                self.norm = torch.nn.BatchNorm1d(3)

            def forward(self, x):
                return self.norm(x)

        model = Normalization()
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    def test_BatchNorm2d(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        class Normalization(torch.nn.Module):
            def __init__(self):
                super(Normalization, self).__init__()
                self.norm = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                return self.norm(x)

        model = Normalization()
        x = torch.rand(shape)
        Tester(request.node.name, model, x)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(
        [
            "-p",
            "no:warnings",
            "-v",
            "test/converter/op_test/caffe/normalization/test_normalization.py",
        ]
    )
