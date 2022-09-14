import os
import sys
import torch
import pytest
import warnings

from brocolli.testing.common_utils import CaffeBaseTester as Tester


def test_ReLU(shape=[1, 3, 32, 32], opset_version=13):
    model = torch.nn.ReLU()
    Tester("ReLU", model, shape, opset_version)


def test_LeakyReLU(shape=[1, 3, 32, 32], opset_version=13):
    model = torch.nn.LeakyReLU()
    Tester("LeakyReLU", model, shape, opset_version)


def test_PReLU(shape=[1, 3, 32, 32], opset_version=13):
    model = torch.nn.PReLU()
    Tester("PReLU", model, shape, opset_version)


def test_Sigmoid(shape=[1, 3, 32, 32], opset_version=13):
    model = torch.nn.Sigmoid()
    Tester("Sigmoid", model, shape, opset_version)


def test_Softmax_basic(shape=[1, 3, 32, 32], opset_version=13):
    model = torch.nn.Softmax()
    Tester("Softmax_basic", model, shape, opset_version)


def test_Softmax_dim_2(shape=[1, 3, 32, 32], opset_version=13):
    model = torch.nn.Softmax(dim=2)
    Tester("Softmax_dim_2", model, shape, opset_version)


def test_Softmax_dim_3(shape=[1, 3, 32, 32], opset_version=13):
    model = torch.nn.Softmax(dim=2)
    Tester("Softmax_dim_3", model, shape, opset_version)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(
        ["-p", "no:warnings", "-v", "test/op_test/caffe/activation/test_activation.py"]
    )
