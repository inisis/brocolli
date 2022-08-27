import torch
import torch.nn as nn
import pytest
import warnings

from bin.converter.utils import OnnxBaseTester as Tester

def test_Conv2d_basic(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=1, stride=1, padding=0)
    Tester("Conv2d_basic", model, shape, opset_version)

def test_Conv2d_kernel_3x3(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=(3, 3), stride=1, padding=0)
    Tester("Conv2d_kernel_3x3", model, shape, opset_version)

def test_Conv2d_kernel_1x3(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=(1, 3), stride=1, padding=0)
    Tester("Conv2d_kernel_1x3", model, shape, opset_version)

def test_Conv2d_kernel_3x1(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=(3, 1), stride=1, padding=0)
    Tester("Conv2d_kernel_3x1", model, shape, opset_version)

def test_Conv2d_stride_3x3(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=1, stride=(3, 3), padding=0)
    Tester("Conv2d_stride_3x3", model, shape, opset_version)

def test_Conv2d_stride_3x1(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=1, stride=(3, 1), padding=0)
    Tester("Conv2d_stride_3x1", model, shape, opset_version)

def test_Conv2d_stride_1x3(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=1, stride=(1, 3), padding=0)
    Tester("Conv2d_stride_1x3", model, shape, opset_version)

def test_Conv2d_padding_3x3(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=1, stride=1, padding=(3, 3))
    Tester("Conv2d_padding_3x3", model, shape, opset_version)

def test_Conv2d_padding_3x1(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=1, stride=1, padding=(3, 1))
    Tester("Conv2d_padding_3x1", model, shape, opset_version)

def test_Conv2d_padding_1x3(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=1, stride=1, padding=(1, 3))
    Tester("Conv2d_padding_1x3", model, shape, opset_version)

def test_Conv2d_dilation_3x3(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=1, stride=1, padding=0, dilation=(3, 3))
    Tester("Conv2d_dilation_3x3", model, shape, opset_version)

def test_Conv2d_dilation_3x1(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=1, stride=1, padding=0, dilation=(3, 1))
    Tester("Conv2d_dilation_3x1", model, shape, opset_version)

def test_Conv2d_dilation_1x3(shape = [1, 3, 32, 32], opset_version=13):
    model = torch.nn.Conv2d(3, 5, kernel_size=1, stride=1, padding=0, dilation=(1, 3))
    Tester("Conv2d_dilation_1x3", model, shape, opset_version)    

def test_Conv1d_module(shape = [1, 3, 32], opset_version=13):
    class Conv1d(torch.nn.Module):
        def __init__(self,in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True):
            super(Conv1d,self).__init__()
            self.conv = nn.Conv1d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias)

        def forward(self,x):
            return self.conv(x)
    model = Conv1d()
    Tester("Conv1d_module", model, shape, opset_version)    

class Conv1d(nn.Module):
    def __init__(self,in_channels=3,out_channels=3,kernel_size=(3,3),stride=(1,1),padding=(1,1),dilation=(1,1),groups=1,bias=True):
        super(Conv1d,self).__init__()
        self.conv = nn.Conv1d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias)

    def forward(self,x):
        return self.conv(x)

def test_Conv1d_basic(shape = [18, 5, 39], opset_version=13):
    model = Conv1d(in_channels=5,out_channels=25,kernel_size=4,stride=2,padding=2,dilation=1,groups=1,bias=False)
    Tester("Conv1d_basic", model, shape, opset_version)    

class ConvTran1d(nn.Module):
    def __init__(self,in_channels=16,out_channels=32,kernel_size=(3,3),stride=(1,1),padding=(1,1),dilation=(1,1),groups=1,bias=True):
        super(ConvTran1d,self).__init__()
        self.conv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride ,
                                       padding=padding, groups=groups, bias=bias, dilation=dilation, padding_mode='zeros')

    def forward(self,x):
        return self.conv(x)

def test_ConvTranspose1d_basic(shape = [18, 5, 39], opset_version=13):
    model = ConvTran1d(in_channels=5,out_channels=25,kernel_size=4,stride=2,padding=2,dilation=1,groups=1,bias=False)
    Tester("ConvTranspose1d_basic", model, shape, opset_version)   


class ConvTran2d(nn.Module):
    def __init__(self,in_channels=16,out_channels=32,kernel_size=(3,3),stride=(1,1),padding=(1,1),dilation=(1,1),groups=1,bias=True):
        super(ConvTran2d,self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride ,
                                       padding=padding, groups=groups, bias=bias, dilation=dilation, padding_mode='zeros')

    def forward(self,x):
        return self.conv(x)

def test_ConvTranspose2d_basic(shape = [18, 5, 24, 24], opset_version=13):
    model = ConvTran2d(in_channels=5,out_channels=25,kernel_size=4,stride=2,padding=2,dilation=1,groups=1,bias=False)
    Tester("ConvTranspose2d_basic", model, shape, opset_version)   

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pytest.main(['-p', 'no:warnings', '-v', 'test/op_test/caffe/conv/test_conv.py'])
