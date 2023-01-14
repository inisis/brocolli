import os
import sys
import torch
import pytest
import warnings
from brocolli.testing.common_utils import CaffeBaseTester as Tester


class TestActivationClass:
    def test_ReLU(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        model = torch.nn.ReLU()
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    def test_LeakyReLU(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        model = torch.nn.LeakyReLU()
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    def test_PReLU(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        model = torch.nn.PReLU()
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    def test_Sigmoid(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        model = torch.nn.Sigmoid()
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    @pytest.mark.parametrize("dim", (-1, 1, 2))
    def test_Softmax(
        self,
        request,
        dim,
        shape=[1, 3, 32, 32],
    ):
        model = torch.nn.Softmax()
        x = torch.rand(shape)
        Tester(request.node.name, model, x)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(
        [
            "-p",
            "no:warnings",
            "-v",
            "test/converter/op_test/caffe/activation/test_activation.py",
        ]
    )
