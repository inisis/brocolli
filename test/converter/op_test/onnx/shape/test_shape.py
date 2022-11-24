import torch
import pytest
import warnings

from brocolli.testing.common_utils import OnnxBaseTester as Tester


class TorchChunk(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(TorchChunk, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        print(torch.chunk(x, *self.args, **self.kwargs))
        return torch.chunk(x, *self.args, **self.kwargs)


class TensorChunk(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(TensorChunk, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return x.chunk(*self.args, **self.kwargs)


def test_TorchChunk_1x1(shape=[1, 3, 3, 3]):
    model = TorchChunk(1, 1)
    Tester("TorchChunk_1x1", model, shape)


def test_TorchChunk_2x1(shape=[1, 3, 3, 3]):
    model = TorchChunk(2, 1)
    Tester("TorchChunk_2x1", model, shape)


def test_TensorChunk_1x1(shape=[1, 3, 3, 3]):
    model = TensorChunk(1, 1)
    Tester("TensorChunk_1x1", model, shape)


def test_TensorChunk_2x1(shape=[1, 3, 3, 3]):
    model = TensorChunk(2, 1)
    Tester("TorchChunk_2x1", model, shape)


class Cat(torch.nn.Module):
    def __init__(self, dim):
        super(Cat, self).__init__()
        self.dim = dim

    def forward(self, x, y, z):
        return torch.cat([x, y, z], dim=self.dim)


def test_Cat(
    shape=((1, 4, 4), (1, 3, 4), (1, 17, 4)),
):
    model = Cat(1)
    Tester("Cat", model, shape)


def test_Cat_neg_dim(
    shape=((1, 4, 4), (1, 3, 4), (1, 17, 4)),
):
    model = Cat(-2)
    Tester("Cat_neg_dim", model, shape)


class Permute(torch.nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.args = args

    def forward(self, x):
        return x.permute(*self.args).contiguous()


def test_Permute_0231(
    shape=[1, 3, 32, 32],
):
    model = Permute(0, 2, 3, 1)
    Tester("Permute_0123", model, shape)


def test_Permute_0312(
    shape=[1, 3, 32, 32],
):
    model = Permute(0, 3, 1, 2)
    Tester("Permute_0123", model, shape)


def test_Permute_04132(
    shape=[1, 2, 3, 4, 5],
):
    model = Permute(0, 4, 1, 3, 2)
    Tester("Permute_04132", model, shape)


class TorchSplit(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(TorchSplit, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return torch.split(x, *self.args, **self.kwargs)


class TensorSplit(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(TensorSplit, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return x.split(*self.args, **self.kwargs)


def test_TorchSplit_1x1(shape=[1, 3, 3, 3]):
    model = TorchSplit(1, 1)
    Tester("TorchSplit_1x1", model, shape)


def test_TorchSplit_2x1(shape=[1, 3, 3, 3]):
    model = TorchSplit(2, 1)
    Tester("TorchSplit_2x1", model, shape)


def test_TensorSplit_1x1(shape=[1, 3, 3, 3]):
    model = TorchSplit(1, 1)
    Tester("TensorSplit_1x1", model, shape)


def test_TensorSplit_2x1(shape=[1, 3, 3, 3]):
    model = TorchSplit(2, 1)
    Tester("TensorSplit_2x1", model, shape)


class Transpose(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Transpose, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        print(self.args)
        return torch.transpose(x, *self.args).contiguous()


def test_Transpose_basic(shape=([1, 3, 3, 3])):
    model = Transpose(1, 2)
    Tester("Transpose_basic", model, shape)


class View(torch.nn.Module):
    def __init__(self, *dims):
        super(View, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.view(*self.dims)


def test_View_basic(shape=([1, 3, 3, 3])):
    model = View(1, -1)
    Tester("View_basic", model, shape)


def test_View_3d(shape=([1, 3, 3, 3])):
    model = View(1, 1, -1)
    Tester("View_3d", model, shape)


def test_View_4d(shape=([1, 3, 3, 3])):
    model = View(1, 3, 3, -1)
    Tester("View_4d", model, shape)


class Tile(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out = torch.tile(x, dims=self.dim)
        return out


def test_tile_basic(shape=([1, 3, 3, 3])):
    model = Tile((1, 1, 2, 1))
    Tester("tile_basic", model, shape)


class Tile_1(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out = torch.tile(x, self.dim)
        return out


def test_tile_1(shape=([1, 3, 3, 3])):
    model = Tile_1((1, 1, 2, 1))
    Tester("tile_1", model, shape)


class Flatten(torch.nn.Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.flatten = torch.nn.Flatten(start_dim, end_dim)

    def forward(self, x):
        return self.flatten(x)


def test_Flatten(shape=([1, 3, 32, 32])):
    model = Flatten()
    Tester("Flatten", model, shape)


def test_Flatten_2(shape=([1, 3, 32, 32])):
    model = Flatten(1, 2)
    Tester("Flatten_2", model, shape)


def test_Flatten_3(shape=([1, 3, 32, 32])):
    model = Flatten(1, -2)
    Tester("Flatten_3", model, shape)


def test_Flatten_4(shape=([1, 3, 32, 32])):
    model = Flatten(0, 2)
    Tester("Flatten_3", model, shape)


class Reshape(torch.nn.Module):
    def __init__(self, shape=None):
        super(Reshape, self).__init__()
        self.shapes = shape

    def forward(self, x):
        return x.reshape(*self.shapes)


def test_Reshape(shape=([1, 3, 32, 32])):
    model = Reshape((-1,))
    Tester("Reshape", model, shape)


def test_Pad(shape=([1, 3, 32, 32])):
    model = torch.nn.ConstantPad1d(2, 0)
    Tester("Pad", model, shape)


def test_Pad2d(shape=([1, 3, 32, 32])):
    model = torch.nn.ConstantPad1d(2, 0)
    Tester("Pad2d", model, shape)


def test_Pad3d(shape=([1, 3, 32, 32])):
    model = torch.nn.ConstantPad3d(3, 1)
    Tester("Pad3d", model, shape)


class Pad1d(torch.nn.Module):
    def __init__(self, padding=None):
        super(Pad1d, self).__init__()
        self.pad = torch.nn.ConstantPad1d(padding, 1)

    def forward(self, x):
        return self.pad(x)


def test_Pad1d_module(shape=([1, 3, 32, 32])):
    model = Pad1d(3)
    Tester("Pad1d_module", model, shape)


class Pad2d(torch.nn.Module):
    def __init__(self, padding=None):
        super(Pad2d, self).__init__()
        self.pad = torch.nn.ConstantPad2d(padding, 0)

    def forward(self, x):
        return self.pad(x)


def test_Pad2d_module(shape=([1, 3, 32, 32])):
    model = Pad2d(3)
    Tester("Pad2d_module", model, shape)


class Pad3d(torch.nn.Module):
    def __init__(self, padding=None):
        super(Pad3d, self).__init__()
        self.pad = torch.nn.ConstantPad3d(padding, 0)

    def forward(self, x):
        return self.pad(x)


def test_Pad3d_module(shape=([1, 3, 32, 32])):
    model = Pad3d(3)
    Tester("Pad3d_module", model, shape)


def test_ZeroPad(shape=([1, 3, 32, 32])):
    model = torch.nn.ZeroPad2d(3)
    Tester("Pad3d", model, shape)


class ZeroPad(torch.nn.Module):
    def __init__(self, padding=None):
        super(ZeroPad, self).__init__()
        self.pad = torch.nn.ZeroPad2d(padding)

    def forward(self, x):
        return self.pad(x)


def test_ZeroPad_module(shape=([1, 3, 32, 32])):
    model = ZeroPad(3)
    Tester("ZeroPad_module", model, shape)


def test_ReflectionPad(shape=([1, 3, 32, 32])):
    model = torch.nn.ReflectionPad2d(3)
    Tester("ReflectionPad2d", model, shape)


class ReflectionPad(torch.nn.Module):
    def __init__(self, padding=None):
        super(ReflectionPad, self).__init__()
        self.pad = torch.nn.ReflectionPad2d(padding)

    def forward(self, x):
        return self.pad(x)


def test_ReflectionPad_module(shape=([1, 3, 32, 32])):
    model = ReflectionPad(3)
    Tester("ReflectionPad_module", model, shape)


def test_ReflectionPad(shape=([1, 3, 32, 32])):
    model = torch.nn.ReplicationPad2d(3)
    Tester("ReflectionPad2d", model, shape)


class ReplicationPad(torch.nn.Module):
    def __init__(self, padding=None):
        super(ReplicationPad, self).__init__()
        self.pad = torch.nn.ReplicationPad2d(padding)

    def forward(self, x):
        return self.pad(x)


def test_ReplicationPad_module(shape=([1, 3, 32, 32])):
    model = ReplicationPad(3)
    Tester("ReplicationPad_module", model, shape)


class Slice(torch.nn.Module):
    def __init__(
        self,
    ):
        super(Slice, self).__init__()

    def forward(self, x):
        return x[:, :, :, 2:3]


def test_Slice_module(shape=([1, 3, 32, 32])):
    model = Slice()
    Tester("Slice_module", model, shape)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(["-p", "no:warnings", "-v", "test/op_test/onnx/shape/test_shape.py"])
