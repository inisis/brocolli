import torch
import torch.nn as nn
import pytest
import warnings
import itertools

from brocolli.testing.common_utils import OnnxBaseTester as Tester


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
        shape=(1, 3, 32, 32),
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

    @pytest.mark.parametrize("kernel_size", (1, 3))
    @pytest.mark.parametrize("stride", (1, 3))
    @pytest.mark.parametrize("dilation", (1, 3))
    def test_Conv1d(
        self,
        request,
        kernel_size,
        stride,
        dilation,
        shape=(1, 3, 32),
    ):
        class Conv1d(torch.nn.Module):
            def __init__(
                self,
                in_channels=3,
                out_channels=3,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
                dilation=dilation,
                groups=1,
                bias=True,
            ):
                super(Conv1d, self).__init__()
                self.conv = nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                )

            def forward(self, x):
                return self.conv(x)

        x = torch.rand(shape)
        model = Conv1d()
        Tester(request.node.name, model, x)

    @pytest.mark.parametrize("kernel_size", (1, 3))
    @pytest.mark.parametrize("stride", (1, 3))
    @pytest.mark.parametrize("dilation", (1, 3))
    def test_ConvTranspose1d_basic(
        self,
        request,
        kernel_size,
        stride,
        dilation,
        shape=(18, 5, 39),
    ):
        class ConvTran1d(nn.Module):
            def __init__(
                self,
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                bias=True,
            ):
                super(ConvTran1d, self).__init__()
                self.conv = nn.ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=bias,
                    dilation=dilation,
                    padding_mode="zeros",
                )

            def forward(self, x):
                return self.conv(x)

        model = ConvTran1d(
            in_channels=5,
            out_channels=25,
            kernel_size=kernel_size,
            stride=stride,
            padding=2,
            dilation=dilation,
            groups=1,
            bias=False,
        )

        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    @pytest.mark.parametrize(
        "kernel_h, kernel_w", list(itertools.product((1, 3), (1, 3)))
    )
    @pytest.mark.parametrize(
        "stride_h, stride_w", list(itertools.product((1, 3), (1, 3)))
    )
    @pytest.mark.parametrize("dilation_h, dilation_w", ((1, 1), (3, 3)))
    def test_ConvTranspose2d_basic(
        self,
        request,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        shape=(18, 5, 24, 24),
    ):
        class ConvTran2d(nn.Module):
            def __init__(
                self,
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                groups=1,
                bias=True,
            ):
                super(ConvTran2d, self).__init__()
                self.conv = nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=bias,
                    dilation=dilation,
                    padding_mode="zeros",
                )

            def forward(self, x):
                return self.conv(x)

        model = ConvTran2d(
            in_channels=5,
            out_channels=25,
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
            padding=2,
            dilation=(dilation_h, dilation_w),
            groups=1,
            bias=False,
        )

        x = torch.rand(shape)
        Tester(request.node.name, model, x)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(
        ["-p", "no:warnings", "-v", "test/converter/op_test/onnx/conv/test_conv.py"]
    )
