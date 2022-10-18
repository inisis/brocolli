import os
import sys
import torch
import pytest
import warnings

from brocolli.testing.common_utils import CaffeBaseTester as Tester


class TestPoolingClass:
    def test_AdaptiveAvgPool2d_1x1(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        model = torch.nn.AdaptiveAvgPool2d((1, 1))
        Tester(request.node.name, model, shape)

    def test_AdaptiveAvgPool2d_1x2(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        model = torch.nn.AdaptiveAvgPool2d((1, 2))
        Tester(request.node.name, model, shape)

    def test_AdaptiveAvgPool2d_2x1(
        self,
        request,
        shape=[1, 3, 32, 32],
    ):
        model = torch.nn.AdaptiveAvgPool2d((2, 1))
        Tester(request.node.name, model, shape)

    def test_AvgPool2d_without_ceil_mode(
        self,
        request,
        shape=[1, 1, 32, 32],
    ):
        model = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        Tester(request.node.name, model, shape)

    def test_AvgPool2d_with_ceil_mode(
        self,
        request,
        shape=[1, 1, 32, 32],
    ):
        model = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        Tester(request.node.name, model, shape)

    def test_MaxPool2d_with_return_indices(
        self,
        request,
        shape=[1, 1, 32, 32],
    ):
        model = torch.nn.MaxPool2d(2, 2, return_indices=True)
        Tester(request.node.name, model, shape)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(
        ["-p", "no:warnings", "-v", "test/op_test/caffe/pooling/test_pooling.py"]
    )
