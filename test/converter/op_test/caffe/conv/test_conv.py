import os
import sys
import torch
import pytest
import warnings
import itertools

from brocolli.testing.common_utils import CaffeBaseTester as Tester


class TestConvClass:
    @pytest.mark.parametrize(
        "kernel_h, kernel_w", list(itertools.product((1, 3), (1, 3)))
    )
    @pytest.mark.parametrize(
        "stride_h, stride_w", list(itertools.product((1, 3), (1, 3)))
    )
    @pytest.mark.parametrize("dilation_h, dilation_w", ((1, 1), (3, 3)))
    def test_Conv2d(
        self,
        request,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        shape=[1, 3, 32, 32],
    ):
        model = torch.nn.Conv2d(
            in_channels=3,
            out_channels=5,
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
            dilation=(dilation_h, dilation_w),
        )
        x = torch.rand(shape)
        Tester(request.node.name, model, x)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(
        ["-p", "no:warnings", "-v", "test/converter/op_test/caffe/conv/test_conv.py"]
    )
