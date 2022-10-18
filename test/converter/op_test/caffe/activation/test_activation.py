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
        Tester(request.node.name, model, shape)

    def test_LeakyReLU(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        model = torch.nn.LeakyReLU()
        Tester(request.node.name, model, shape)

    def test_PReLU(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        model = torch.nn.PReLU()
        Tester(request.node.name, model, shape)

    def test_Sigmoid(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        model = torch.nn.Sigmoid()
        Tester(request.node.name, model, shape)

    def test_Softmax_basic(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        model = torch.nn.Softmax()
        Tester(request.node.name, model, shape)

    def test_Softmax_dim_2(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        model = torch.nn.Softmax(dim=2)
        Tester(request.node.name, model, shape)

    def test_Softmax_dim_3(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        model = torch.nn.Softmax(dim=2)
        Tester(request.node.name, model, shape)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(
        ["-p", "no:warnings", "-v", "test/op_test/caffe/activation/test_activation.py"]
    )
