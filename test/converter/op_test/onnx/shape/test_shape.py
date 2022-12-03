import torch
import pytest
import warnings

from brocolli.testing.common_utils import OnnxBaseTester as Tester


class TestShapeClass:
    @pytest.mark.parametrize("chunks", (1, 2))
    @pytest.mark.parametrize("dim", (1, 2))
    def test_TorchChunk(self, request, chunks, dim, shape=[1, 3, 3, 3]):
        class TorchChunk(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super(TorchChunk, self).__init__()
                self.args = args
                self.kwargs = kwargs

            def forward(self, x):
                out1 = torch.chunk(x, *self.args, **self.kwargs)
                out2 = x.chunk(*self.args, **self.kwargs)
                return out1, out2

        model = TorchChunk(chunks, dim)
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
        y = torch.rand(shape[0])
        z = torch.rand(shape[0])
        Tester(request.node.name, model, (x, y, z))

    @pytest.mark.parametrize("order", ((0, 2, 3, 1), (0, 3, 1, 2)))
    def test_Permute(
        self,
        request,
        order,
        shape=(1, 3, 32, 32),
    ):
        class Permute(torch.nn.Module):
            def __init__(self, *args):
                super(Permute, self).__init__()
                self.args = args

            def forward(self, x):
                return x.permute(*self.args).contiguous()

        model = Permute(order)
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    @pytest.mark.parametrize("section", (1, 2))
    @pytest.mark.parametrize("dim", (1, 2))
    def test_TorchSplit_1x1(self, request, section, dim, shape=(1, 3, 3, 3)):
        class TorchSplit(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super(TorchSplit, self).__init__()
                self.args = args
                self.kwargs = kwargs

            def forward(self, x):
                return torch.split(x, *self.args, **self.kwargs) + x.split(
                    *self.args, **self.kwargs
                )

        model = TorchSplit(section, dim)
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    def test_Transpose_basic(self, request, shape=([1, 3, 3, 3])):
        class Transpose(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super(Transpose, self).__init__()
                self.args = args
                self.kwargs = kwargs

            def forward(self, x):
                return torch.transpose(x, *self.args).contiguous()

        model = Transpose(1, 2)
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    def test_tile(self, request, shape=([1, 3, 3, 3])):
        class Tile(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                out = torch.tile(x, dims=self.dim)
                return out

        model = Tile((1, 1, 2, 1))
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    @pytest.mark.parametrize("start_dim", (0, 1))
    @pytest.mark.parametrize("end_dim", (1, 2, -2))
    def test_Flatten(self, request, start_dim, end_dim, shape=([1, 3, 32, 32])):
        class Flatten(torch.nn.Module):
            def __init__(self, start_dim=1, end_dim=-1):
                super(Flatten, self).__init__()
                self.flatten = torch.nn.Flatten(start_dim, end_dim)

            def forward(self, x):
                return self.flatten(x)

        model = Flatten(start_dim, end_dim)
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    def test_Reshape(self, request, shape=([1, 3, 32, 32])):
        class Reshape(torch.nn.Module):
            def __init__(self, shape=None):
                super(Reshape, self).__init__()
                self.shapes = shape

            def forward(self, x):
                return x.reshape(*self.shapes)

        model = Reshape((-1,))
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    def test_Slice_module(self, request, shape=([1, 3, 32, 32])):
        class Slice(torch.nn.Module):
            def __init__(self):
                super(Slice, self).__init__()

            def forward(self, x):
                return x[:, :, :, 2:3]

        model = Slice()
        x = torch.rand(shape)
        Tester(request.node.name, model, x)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(
        ["-p", "no:warnings", "-v", "test/converter/op_test/onnx/shape/test_shape.py"]
    )
