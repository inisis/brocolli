import torch
import torchvision
import pytest
import warnings

from brocolli.testing.common_utils import OnnxBaseTester as Tester


class TestPoolingClass:
    @pytest.mark.parametrize("output_size_h, output_size_w", ((1, 1), (4, 4)))
    def test_AdaptiveAvgPool2d(
        self,
        request,
        output_size_h,
        output_size_w,
        shape=(1, 3, 32, 32),
    ):
        model = torch.nn.AdaptiveAvgPool2d((output_size_h, output_size_w))
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    def test_AdaptiveAvgPool1d(
        self,
        request,
        shape=(1, 3, 32),
    ):
        model = torch.nn.AdaptiveAvgPool1d((1))
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    def test_AvgPool2d_without_ceil_mode(
        self,
        request,
        shape=(1, 1, 32, 32),
    ):
        model = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    def test_AvgPool2d_with_ceil_mode(
        self,
        request,
        shape=(1, 1, 32, 32),
    ):
        model = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    def test_AvgPool1d(
        self,
        request,
        shape=(1, 1, 32),
    ):
        model = torch.nn.AvgPool1d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    def test_AvgPool1d_Module(
        self,
        request,
        shape=(1, 1, 32),
    ):
        class AvgPool1d(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = torch.nn.AvgPool1d(
                    kernel_size=3, stride=2, padding=1, ceil_mode=True
                )

            def forward(self, x):
                return self.pool(x)

        model = AvgPool1d()
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    def test_AvgPool2d_Module(
        self,
        request,
        shape=(1, 1, 32, 32),
    ):
        class AvgPool2d(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = torch.nn.AvgPool2d(
                    kernel_size=3, stride=2, padding=1, ceil_mode=True
                )

            def forward(self, x):
                return self.pool(x)

        model = AvgPool2d()
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    def test_MaxPool1d(
        self,
        request,
        shape=(1, 1, 32),
    ):
        model = torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    def test_MaxPool1d_Module(
        self,
        request,
        shape=(1, 1, 32),
    ):
        class MaxPool1d(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = torch.nn.MaxPool1d(
                    kernel_size=3, stride=2, padding=1, ceil_mode=True
                )

            def forward(self, x):
                return self.pool(x)

        model = MaxPool1d()
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    def test_MaxPool2d_Module(
        self,
        request,
        shape=(1, 1, 32, 32),
    ):
        class MaxPool2d(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = torch.nn.MaxPool2d(
                    kernel_size=3, stride=2, padding=1, ceil_mode=True
                )

            def forward(self, x):
                return self.pool(x)

        model = MaxPool2d()
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    @pytest.mark.parametrize("output_size", (1, 4))
    def test_AdaptiveAvgPool1d_Module(
        self,
        request,
        output_size,
        shape=(1, 1, 32),
    ):
        class AdaptiveAvgPool1d(torch.nn.Module):
            def __init__(self, output_size):
                super().__init__()
                self.pool = torch.nn.AdaptiveAvgPool1d(output_size)

            def forward(self, x):
                return self.pool(x)

        model = AdaptiveAvgPool1d(output_size)
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    @pytest.mark.parametrize("output_size_h, output_size_w", ((1, 1), (4, 4)))
    def test_AdaptiveAvgPool2d_Module(
        self,
        request,
        output_size_h,
        output_size_w,
        shape=(1, 1, 32, 32),
    ):
        class AdaptiveAvgPool2d(torch.nn.Module):
            def __init__(self, output_size_h, output_size_w):
                super().__init__()
                self.pool = torch.nn.AdaptiveAvgPool2d((output_size_h, output_size_w))

            def forward(self, x):
                return self.pool(x)

        model = AdaptiveAvgPool2d(output_size_h, output_size_w)
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    @pytest.mark.parametrize("scale_factor", (2, 64))
    def test_Upsample(
        self,
        request,
        scale_factor,
        shape=(1, 1, 32, 32),
    ):
        model = torch.nn.Upsample(scale_factor=scale_factor)
        x = torch.rand(shape)
        Tester(request.node.name, model, x)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(
        [
            "-p",
            "no:warnings",
            "-v",
            "test/converter/op_test/onnx/pooling/test_pooling.py",
        ]
    )
