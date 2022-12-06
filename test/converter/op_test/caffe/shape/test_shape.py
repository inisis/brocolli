import os
import sys
import torch
import pytest
import warnings

from brocolli.testing.common_utils import CaffeBaseTester as Tester


class TestShapeClass:
    @pytest.mark.parametrize("chunks", (1, 2))
    def test_TorchChunk_1x1(self, request, chunks, shape=[1, 3, 3, 3]):
        class TorchChunk(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super(TorchChunk, self).__init__()
                self.args = args
                self.kwargs = kwargs

            def forward(self, x):
                return torch.chunk(x, *self.args, **self.kwargs)

        model = TorchChunk(chunks, dim=1)
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    @pytest.mark.parametrize("chunks", (1, 2))
    def test_TensorChunk_1x1(self, request, chunks, shape=[1, 3, 3, 3]):
        class TensorChunk(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super(TensorChunk, self).__init__()
                self.args = args
                self.kwargs = kwargs

            def forward(self, x):
                return x.chunk(*self.args, **self.kwargs)

        model = TensorChunk(1, dim=1)
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    @pytest.mark.parametrize("dim", (1, -2))
    def test_Cat(
        self,
        request,
        dim,
        shape=((1, 4, 4), (1, 3, 4), (1, 17, 4)),
    ):
        class Cat(torch.nn.Module):
            def __init__(self, dim):
                super(Cat, self).__init__()
                self.dim = dim

            def forward(self, x, y, z):
                return torch.cat([x, y, z], dim=self.dim)

        model = Cat(dim)
        x = torch.rand(shape[0])
        y = torch.rand(shape[1])
        z = torch.rand(shape[2])
        Tester(request.node.name, model, (x, y, z))

    @pytest.mark.parametrize("split_size_or_sections", (1, 2))
    def test_TorchSplit(self, request, split_size_or_sections, shape=[1, 3, 3, 3]):
        class TorchSplit(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super(TorchSplit, self).__init__()
                self.args = args
                self.kwargs = kwargs

            def forward(self, x):
                return torch.split(x, *self.args, **self.kwargs)

        model = TorchSplit(split_size_or_sections, dim=1)
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    @pytest.mark.parametrize("split_size", (1, 2))
    def test_TensorSplit(self, request, split_size, shape=[1, 3, 3, 3]):
        class TensorSplit(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super(TensorSplit, self).__init__()
                self.args = args
                self.kwargs = kwargs

            def forward(self, x):
                return x.split(*self.args, **self.kwargs)

        model = TensorSplit(1, 1)
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    @pytest.mark.parametrize(("dims"), [(1, -1), (1, 1, -1), (1, 3, 3, -1)])
    def test_View(self, request, dims, shape=([1, 3, 3, 3])):
        class View(torch.nn.Module):
            def __init__(self, *dims):
                super(View, self).__init__()
                self.dims = dims

            def forward(self, x):
                return x.view(*self.dims)

        model = View(dims)
        x = torch.rand(shape)
        Tester(request.node.name, model, x)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(
        ["-p", "no:warnings", "-v", "test/converter/op_test/caffe/shape/test_shape.py"]
    )
