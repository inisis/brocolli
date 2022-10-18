import torch
import torchvision
import pytest
import warnings

from brocolli.testing.common_utils import OnnxBaseTester as Tester


def test_AdaptiveAvgPool2d_1x1(
    shape=[1, 3, 32, 32],
):
    model = torch.nn.AdaptiveAvgPool2d((1, 1))
    Tester("AdaptiveAvgPool2d_1x1", model, shape)


def test_AdaptiveAvgPool2d_1x2(
    shape=[1, 3, 32, 32],
):
    model = torch.nn.AdaptiveAvgPool2d((1, 2))
    Tester("AdaptiveAvgPool2d_1x2", model, shape)


def test_AdaptiveAvgPool2d_2x1(
    shape=[1, 3, 32, 32],
):
    model = torch.nn.AdaptiveAvgPool2d((2, 1))
    Tester("AdaptiveAvgPool2d_2x1", model, shape)


def test_AdaptiveAvgPool1d_1(
    shape=[1, 3, 32],
):
    model = torch.nn.AdaptiveAvgPool1d((1))
    Tester("AdaptiveAvgPool1d_1", model, shape)


def test_AvgPool2d_without_ceil_mode(
    shape=[1, 1, 32, 32],
):
    model = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
    Tester("AvgPool2d_without_ceil_mode", model, shape)


def test_AvgPool2d_with_ceil_mode(
    shape=[1, 1, 32, 32],
):
    model = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
    Tester("AvgPool2d_with_ceil_mode", model, shape)


def test_AvgPool1d(
    shape=[1, 1, 32],
):
    model = torch.nn.AvgPool1d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
    Tester("AvgPool1d", model, shape)


def test_AvgPool1d_Module(
    shape=[1, 1, 32],
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
    Tester("AvgPool1d_Module", model, shape)


def test_AvgPool2d_Module(
    shape=[1, 1, 32, 32],
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
    Tester("AvgPool2d_Module", model, shape)


def test_MaxPool1d(
    shape=[1, 1, 32],
):
    model = torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
    Tester("MaxPool1d", model, shape)


def test_MaxPool1d_Module(
    shape=[1, 1, 32],
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
    Tester("MaxPool1d_Module", model, shape)


def test_MaxPool2d_Module(
    shape=[1, 1, 32, 32],
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
    Tester("MaxPool2d_Module", model, shape)


def test_AdaptiveAvgPool1d_Module(
    shape=[1, 1, 32],
):
    class AdaptiveAvgPool1d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool1d(1)

        def forward(self, x):
            return self.pool(x)

    model = AdaptiveAvgPool1d()
    Tester("AdaptiveAvgPool1d_Module", model, shape)


def test_AdaptiveAvgPool1d_7_Module(
    shape=[1, 1, 32],
):
    class AdaptiveAvgPool1d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool1d(4)

        def forward(self, x):
            return self.pool(x)

    model = AdaptiveAvgPool1d()
    Tester("AdaptiveAvgPool1d_7_Module", model, shape)


def test_AdaptiveAvgPool2d_Module(
    shape=[1, 1, 32, 32],
):
    class AdaptiveAvgPool2d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            return self.pool(x)

    model = AdaptiveAvgPool2d()
    Tester("AdaptiveAvgPool2d_Module", model, shape)


def test_AdaptiveAvgPool2d_7x7_Module(
    shape=[1, 1, 32, 32],
):
    class AdaptiveAvgPool2d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d((4, 4))

        def forward(self, x):
            return self.pool(x)

    model = AdaptiveAvgPool2d()
    Tester("AdaptiveAvgPool2d_7x7_Module", model, shape)


def test_Upsample(
    shape=[1, 1, 32, 32],
):
    model = torch.nn.Upsample(scale_factor=2)
    Tester("Upsample", model, shape)


def test_Upsample_1(
    shape=[1, 1, 32, 32],
):
    model = torch.nn.Upsample(64)
    Tester("Upsample_1", model, shape)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(
        ["-p", "no:warnings", "-v", "test/op_test/caffe/pooling/test_pooling.py"]
    )
