import os
import sys
import torch
import pytest
import warnings

from bin.utils import CaffeBaseTester as Tester

class TorchChunk(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(TorchChunk, self).__init__()
        self.args = args
        self.kwargs = kwargs
        
    def forward(self, x):
        return torch.chunk(x, *self.args, **self.kwargs)

class TensorChunk(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(TensorChunk, self).__init__()
        self.args = args
        self.kwargs = kwargs
        
    def forward(self, x):
        return x.chunk(*self.args, **self.kwargs)

def test_TorchChunk_1x1(shape = [1, 3, 3, 3], opset_version=9):
    model = TorchChunk(1, 1)
    Tester("TorchChunk_1x1", model, shape, opset_version)

def test_TorchChunk_2x1(shape = [1, 3, 3, 3], opset_version=9):
    model = TorchChunk(2, 1)
    Tester("TorchChunk_2x1", model, shape, opset_version)

def test_TensorChunk_1x1(shape = [1, 3, 3, 3], opset_version=9):
    model = TensorChunk(1, 1)
    Tester("TensorChunk_1x1", model, shape, opset_version)

def test_TensorChunk_2x1(shape = [1, 3, 3, 3], opset_version=9):
    model = TensorChunk(2, 1)
    Tester("TorchChunk_2x1", model, shape, opset_version)    

class Cat(torch.nn.Module):
    def __init__(self, dim):
        super(Cat, self).__init__()
        self.dim = dim

    def forward(self, *x):
        return torch.cat(x, dim=self.dim)

def test_Cat(shape = ((1, 4, 4), (1, 3, 4), (1, 17, 4)) ,opset_version=13):
    model = Cat(1)
    Tester("Cat", model, shape, opset_version)    

def test_Cat_neg_dim(shape = ((1, 4, 4), (1, 3, 4), (1, 17, 4)) ,opset_version=13):
    model = Cat(-2)
    Tester("Cat_neg_dim", model, shape, opset_version)    

def test_Upsample_scale_factor2_nearest(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Upsample(scale_factor=2, mode='nearest')
    Tester("Upsample_scale_factor2_nearest", model, shape, opset_version)

class Interpolate(torch.nn.Module):
    def __init__(self, size=None,scale_factor=None, mode=None, align_corners=None):
        super(Interpolate, self).__init__()
		## Use either size or scale factor. 
        self.size = size
        self.scale_factor = scale_factor 
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return torch.nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor,mode=self.mode, align_corners=self.align_corners)

def test_Interpolate_bilinear(shape = [1, 3, 32, 32], opset_version=9):
    model = Interpolate(scale_factor=8, mode='bilinear', align_corners=None)
    Tester("Interpolate_bilinear", model, shape, opset_version)

def test_Interpolate_bilinear_align_corners(shape = [1, 3, 32, 32], opset_version=9):
    model = Interpolate(scale_factor=8, mode='bilinear', align_corners=True)
    Tester("Interpolate_bilinear_align_cornersalign_corners", model, shape, opset_version)

def test_Interpolate_bilinear_size(shape = [1, 3, 32, 32], opset_version=9):
    model = Interpolate(size=(224, 224), mode='bilinear', align_corners=None)
    Tester("Interpolate_bilinear_size", model, shape, opset_version)

def test_Interpolate_bilinear_size_align_corners(shape = [1, 3, 32, 32], opset_version=9):
    model = Interpolate(size=(224, 224), mode='bilinear', align_corners=True)
    Tester("Interpolate_bilinear_size_align_corners", model, shape, opset_version)

class Pad(torch.nn.Module):
    def __init__(self, pad, mode='constant'):
        super(Pad, self).__init__()
        self.pad = pad
        self.mode = mode
        
    def forward(self, x):
        return torch.nn.functional.pad(x, self.pad, self.mode)

def test_Pad_basic(shape = [1, 3, 32, 32], opset_version=13):
    model = Pad((1, 2, 3, 4))
    Tester("Pad_basic", model, shape, opset_version)    

def test_Pad_reflect(shape = [1, 3, 32, 32], opset_version=13):
    model = Pad((1, 2, 3, 4), 'reflect')
    Tester("Pad_reflect", model, shape, opset_version)

class Permute(torch.nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.args = args
    def forward(self, x):
        return x.permute(*self.args).contiguous()

def test_Permute_0231(shape = [1, 3, 32, 32], opset_version=13):
    model = Permute(0, 2, 3, 1)
    Tester("Permute_0123", model, shape, opset_version)

def test_Permute_0312(shape = [1, 3, 32, 32], opset_version=13):
    model = Permute(0, 3, 1, 2)
    Tester("Permute_0123", model, shape, opset_version)    

def test_Permute_04132(shape = [1, 2, 3, 4, 5], opset_version=13):
    model = Permute(0, 4, 1, 3, 2)
    Tester("Permute_04132", model, shape, opset_version)

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

def test_TorchSplit_1x1(shape = [1, 3, 3, 3], opset_version=9):
    model = TorchSplit(1, 1)
    Tester("TorchSplit_1x1", model, shape, opset_version)

def test_TorchSplit_2x1(shape = [1, 3, 3, 3], opset_version=9):
    model = TorchSplit(2, 1)
    Tester("TorchSplit_2x1", model, shape, opset_version)

def test_TensorSplit_1x1(shape = [1, 3, 3, 3], opset_version=9):
    model = TorchSplit(1, 1)
    Tester("TensorSplit_1x1", model, shape, opset_version)

def test_TensorSplit_2x1(shape = [1, 3, 3, 3], opset_version=9):
    model = TorchSplit(2, 1)
    Tester("TensorSplit_2x1", model, shape, opset_version)

class Stack(torch.nn.Module):
    def __init__(self, dim):
        super(Stack, self).__init__()
        self.dim = dim

    def forward(self, *x):
        return torch.stack(x, dim=self.dim)

def test_Stack_1(shape = ([1, 3, 3, 3], [1, 3, 3, 3]), opset_version=9):
    model = Stack(1)
    Tester("Stack_1", model, shape, opset_version)

def test_Stack_neg_dim1(shape = ([1, 3, 3, 3], [1, 3, 3, 3]), opset_version=9):
    model = Stack(-1)
    Tester("Stack_neg_dim1", model, shape, opset_version)

class Transpose(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Transpose, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        print(self.args)
        return torch.transpose(x, *self.args).contiguous()

def test_Transpose_basic(shape = ([1, 3, 3, 3]), opset_version=9):
    model = Transpose(1, 2)
    Tester("Transpose_basic", model, shape, opset_version)

class View(torch.nn.Module):
    def __init__(self, *dims):
        super(View, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.view(*self.dims)

def test_View_basic(shape = ([1, 3, 3, 3]), opset_version=9):
    model = View(1, -1)
    Tester("View_basic", model, shape, opset_version)

def test_View_3d(shape = ([1, 3, 3, 3]), opset_version=9):
    model = View(1, 1, -1)
    Tester("View_3d", model, shape, opset_version)

def test_View_4d(shape = ([1, 3, 3, 3]), opset_version=9):
    model = View(1, 3, 3, -1)
    Tester("View_4d", model, shape, opset_version)

class UnSqueeze(torch.nn.Module):
    def __init__(self, dim):
        super(UnSqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(dim=self.dim)   

def test_UnSqueeze_basic(shape = ([1, 3, 1, 3]), opset_version=9):
    model = UnSqueeze(2)
    Tester("UnSqueeze_basic", model, shape, opset_version)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pytest.main(['-p', 'no:warnings', '-v', 'test/op_test/caffe/shape/test_shape.py'])
