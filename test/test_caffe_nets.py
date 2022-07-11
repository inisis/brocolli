import os
os.environ[
    "GLOG_minloglevel"
] = "3"  # 0 - debug 1 - info (still a LOT of outputs) 2 - warnings 3 - errors

import pytest
import warnings

import torchvision.models as models

from bin.jit.pytorch2caffe import Runner

FUSE = True

os.makedirs('tmp', exist_ok=True)

def test_alexnet(shape = [1, 3, 224, 224], opset_version=9, fuse=FUSE):
    net = models.alexnet(pretrained=False)
    runner = Runner("alexnet", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_resnet18(shape = [1, 3, 224, 224], opset_version=9, fuse=FUSE):
    net = models.resnet18(pretrained=False)
    runner = Runner("resnet18", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_squeezenet(shape = [1, 3, 227, 227], opset_version=9, fuse=FUSE):
    net = models.squeezenet1_0(pretrained=False)
    runner = Runner("squeezenet", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_googlenet(shape = [1, 3, 224, 224], opset_version=13, fuse=FUSE):
    net = models.googlenet(pretrained=False)
    runner = Runner("googlenet", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_mobilenet_v2(shape = [1, 3, 224, 224], opset_version=9, fuse=FUSE):
    net = models.mobilenet_v2(pretrained=False)
    runner = Runner("mobilenet", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_mobilenet_v3(shape = [1, 3, 224, 224], opset_version=13, fuse=FUSE):
    '''
    symbolic_opset13.py
    @parse_args("v")
    def hardswish(g, self):
        return g.op("HardSwish", self)
    '''
    net = models.mobilenet_v3_small(pretrained=False)
    runner = Runner("mobilenet_v3", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_densenet121(shape = [1, 3, 224, 224], opset_version=9, fuse=FUSE):
    net = models.densenet121(pretrained=False)
    runner = Runner("densenet121", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_densenet161(shape = [1, 3, 224, 224], opset_version=9, fuse=FUSE):
    net = models.densenet161(pretrained=False)
    runner = Runner("densenet161", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_shufflenet(shape = [1, 3, 224, 224], opset_version=9, fuse=FUSE):
    net = models.shufflenet_v2_x1_0(pretrained=False)
    runner = Runner("shufflenet_v2_x1_0", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_yolov5(shape = [1, 3, 640, 640], opset_version=13, fuse=FUSE):
    import torch
    net = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False, pretrained=False, device=torch.device('cpu'))

    class Identity(torch.nn.Module):
        def __init__(self):
            super(Identity, self).__init__()
            
        def forward(self, x):
            for i in range(self.nl):
                x[i] = self.m[i](x[i])
                bs, _, ny, nx = x[i].shape
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
       
            return x
    
    name, _ = list(net.model.named_children())[-1]
    identity = Identity()
    detect = getattr(net.model, name)
    identity.__dict__.update(detect.__dict__)
    setattr(net.model, name, identity)

    runner = Runner("yolov5", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_ssd300_vgg16(shape = [1, 3, 300, 300], opset_version=13, fuse=FUSE):
    '''
    symbolic_opset13.py
    @parse_args('v', 'v', 'v', 'i', 'i', 'i')
    def linalg_norm(g, self):
        return g.op("LpNormalization", self)
    '''    
    from custom_models.ssd import build_ssd
    net = build_ssd("export")
    runner = Runner("ssd300_vgg16", net, shape, opset_version, fuse)
    runner.pyotrch_inference()  
    runner.convert(export_mode=True)
    runner.caffe_inference()
    runner.check_result()

def test_yolov3(shape = [1, 3, 416, 416], opset_version=13, fuse=FUSE):
    '''
    symbolic_helper.py
    size = _maybe_get_const(args[0:][0], 'is')
    
    return g.op("Resize",
                input,
                empty_roi,
                scales,
                coordinate_transformation_mode_s=coordinate_transformation_mode,
                cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
                mode_s=interpolate_mode,  # nearest, linear, or cubic
                nearest_mode_s="floor",
                scale_factor_i = size)  # only valid when mode="nearest"
    '''    
    from custom_models.yolov3 import Darknet
    net = Darknet('custom_models/yolov3.cfg', 416)
    runner = Runner("yolov3", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_scnn(shape = [1, 3, 512, 288], opset_version=9, fuse=FUSE):
    '''
    symbolic_opset9.py    
    def upsample_bilinear2d(g, input, output_size, *args):
        scales, align_corners = sym_help._get_interpolate_attributes(g, "linear", args)
        sym_help._interpolate_warning("linear")
        align_corners = sym_help._maybe_get_scalar(align_corners)    
        if align_corners:
            align_corners_ = True
        else:
            align_corners_ = False
        if scales is None:
            scales = sym_help._interpolate_size_to_scales(g, input, output_size, 4)

    return g.op("BilinearInterpolate", input, scales, mode_s="linear", align_corners_i=align_corners_)    
    '''
    from custom_models.scnn import SCNN
    net = SCNN(input_size=[512, 288], pretrained=False)

    runner = Runner("scnn", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_segnet(shape = [1, 3, 360, 480], opset_version=13, fuse=FUSE):
    '''
    symbolic_opset13.py  
    def max_unpool2d(g, self, indices, output_size):
        return g.op("MaxUnpool", self, indices, output_size)
    
    symbolic_opset10.py
    if return_indices:
        r, indices = g.op("MaxPool", input, outputs=2, **kwargs)
        return r, indices        
    '''
    from custom_models.segnet import SegNet
    net = SegNet()
    runner = Runner("segnet", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()

def test_realcugan(shape = [1, 3, 200, 200], opset_version=13, fuse=FUSE):
    '''
    symbolic_opset11.py
    def constant_pad_nd(g, input, padding, value=None):
        mode = "constant"
        value = sym_help._maybe_get_scalar(value)
        value = sym_help._if_scalar_type_as(g, value, input)
        pad = _prepare_onnx_paddings(g, sym_help._get_tensor_rank(input), padding)
        padding = torch.onnx.symbolic_opset9._convert_padding_node(padding)
        paddings_ = torch.onnx.symbolic_opset9._prepare_onnx_paddings(sym_help._get_tensor_rank(input), padding)   
        return g.op("Pad", input, pad, value, mode_s=mode, pads_i=paddings_)


    def reflection_pad(g, input, padding):
        mode = "reflect"
        paddings = _prepare_onnx_paddings(g, sym_help._get_tensor_rank(input), padding)
        padding = torch.onnx.symbolic_opset9._convert_padding_node(padding)
        paddings_ = torch.onnx.symbolic_opset9._prepare_onnx_paddings(sym_help._get_tensor_rank(input), padding)
        return g.op("Pad", input, paddings, mode_s=mode, pads_i=paddings_)    
    '''
    from custom_models.upcunet_v3 import RealWaifuUpScaler
    upscaler2x = RealWaifuUpScaler(2, "custom_models/up2x-latest-denoise3x.pth",
                                half=False, device="cpu")
 
    runner = Runner("Real_CUGAN", upscaler2x.model, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pytest.main(['-p', 'no:warnings', '-v', 'test/test_caffe_nets.py'])
