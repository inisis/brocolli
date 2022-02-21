import os
os.environ[
    "GLOG_minloglevel"
] = "3"  # 0 - debug 1 - info (still a LOT of outputs) 2 - warnings 3 - errors

import pytest
import warnings

import torchvision.models as models

from bin.pytorch_model_converter import Runner

FUSE = True

# def test_alexnet(shape = [1, 3, 224, 224], opset_version=9, fuse=FUSE):
#     net = models.alexnet(pretrained=False)
#     runner = Runner("alexnet", net, shape, opset_version, fuse)
#     runner.pyotrch_inference()
#     runner.convert()
#     runner.caffe_inference()
#     runner.check_result()

# def test_resnet18(shape = [1, 3, 224, 224], opset_version=9, fuse=FUSE):
#     net = models.resnet18(pretrained=False)
#     runner = Runner("resnet18", net, shape, opset_version, fuse)
#     runner.pyotrch_inference()
#     runner.convert()
#     runner.caffe_inference()
#     runner.check_result()

# def test_squeezenet(shape = [1, 3, 227, 227], opset_version=9, fuse=FUSE):
#     net = models.squeezenet1_0(pretrained=False)
#     runner = Runner("squeezenet", net, shape, opset_version, fuse)
#     runner.pyotrch_inference()
#     runner.convert()
#     runner.caffe_inference()
#     runner.check_result()

# def test_googlenet(shape = [1, 3, 224, 224], opset_version=9, fuse=FUSE):
#     net = models.googlenet(pretrained=False)
#     runner = Runner("googlenet", net, shape, opset_version, fuse)
#     runner.pyotrch_inference()
#     runner.convert()
#     runner.caffe_inference()
#     runner.check_result()

# def test_mobilenet_v2(shape = [1, 3, 224, 224], opset_version=9, fuse=FUSE):
#     net = models.mobilenet_v2(pretrained=False)
#     runner = Runner("mobilenet", net, shape, opset_version, fuse)
#     runner.pyotrch_inference()
#     runner.convert()
#     runner.caffe_inference()
#     runner.check_result()

# def test_mobilenet_v3(shape = [1, 3, 224, 224], opset_version=13, fuse=FUSE):
#     '''
#     symbolic_opset13.py
#     @parse_args("v")
#     def hardswish(g, self):
#         return g.op("HardSwish", self)
#     '''
#     net = models.mobilenet_v3_small(pretrained=False)
#     runner = Runner("mobilenet_v3", net, shape, opset_version, fuse)
#     runner.pyotrch_inference()
#     runner.convert()
#     runner.caffe_inference()
#     runner.check_result()

# def test_densenet121(shape = [1, 3, 224, 224], opset_version=9, fuse=FUSE):
#     net = models.densenet121(pretrained=False)
#     runner = Runner("densenet121", net, shape, opset_version, fuse)
#     runner.pyotrch_inference()
#     runner.convert()
#     runner.caffe_inference()
#     runner.check_result()

# def test_densenet161(shape = [1, 3, 224, 224], opset_version=9, fuse=FUSE):
#     net = models.densenet161(pretrained=False)
#     runner = Runner("densenet161", net, shape, opset_version, fuse)
#     runner.pyotrch_inference()
#     runner.convert()
#     runner.caffe_inference()
#     runner.check_result()   

# def test_inception_v3(shape = [1, 3, 299, 299], opset_version=9, fuse=FUSE):
#     net = models.inception_v3(pretrained=False)
#     runner = Runner("inception_v3", net, shape, opset_version, fuse)
#     runner.pyotrch_inference()
#     runner.convert()
#     runner.caffe_inference()
#     runner.check_result()  

# def test_vgg16(shape = [1, 3, 224, 224], opset_version=9, fuse=FUSE):
#     net = models.vgg16(pretrained=False)
#     runner = Runner("vgg16", net, shape, opset_version, fuse)
#     runner.pyotrch_inference()
#     runner.convert()
#     runner.caffe_inference()
#     runner.check_result()

# def test_shufflenet(shape = [1, 3, 224, 224], opset_version=9, fuse=FUSE):
#     '''
#     shufflenetv2.py
#     def channel_shuffle(x: Tensor, groups: int) -> Tensor:
#         # reshape
#         x = x.view(int(x.size(0)), groups, -1, int(x.size(2)), int(x.size(3)))
#         x = torch.transpose(x, 1, 2).contiguous()
#         # flatten
#         x = x.view(int(x.size(0)), -1, int(x.size(3)), int(x.size(4)))
#         return x
#     '''       
#     net = models.shufflenet_v2_x1_0(pretrained=False)
#     runner = Runner("shufflenet_v2_x1_0", net, shape, opset_version, fuse)
#     runner.pyotrch_inference()
#     runner.convert()
#     runner.caffe_inference()
#     runner.check_result()

# def test_yolov5(shape = [1, 3, 640, 640], opset_version=13, fuse=FUSE):
#     '''
#     def parse_model(d, ch):

#     elif m is Detect:
#         continue
    
#     '''

#     import torch
#     net = torch.hub.load('ultralytics/yolov5', 'yolov5l', autoshape=False, pretrained=False, device=torch.device('cpu'))
#     runner = Runner("yolov5", net, shape, opset_version, fuse)
#     runner.pyotrch_inference()
#     runner.convert()
#     runner.caffe_inference()
#     runner.check_result()

# def test_ssd300_vgg16(shape = [1, 3, 300, 300], opset_version=13, fuse=FUSE):
#     '''
#     symbolic_opset13.py
#     @parse_args('v', 'v', 'v', 'i', 'i', 'i')
#     def linalg_norm(g, self):
#         return g.op("LpNormalization", self)
#     '''    
#     from custom_models.ssd import build_ssd
#     net = build_ssd("export")
#     runner = Runner("ssd300_vgg16", net, shape, opset_version, fuse)
#     runner.pyotrch_inference()  
#     runner.convert(export_mode=True)
#     runner.caffe_inference()
#     runner.check_result()

# def test_yolov3(shape = [1, 3, 416, 416], opset_version=13, fuse=FUSE):
#     from custom_models.yolov3 import Darknet
#     net = Darknet('custom_models/yolov3.cfg', 416)
#     runner = Runner("yolov3", net, shape, opset_version, fuse)
#     runner.pyotrch_inference()
#     runner.convert()
#     runner.caffe_inference()
#     runner.check_result()

# def test_scnn(shape = [1, 3, 512, 288], opset_version=9, fuse=FUSE):
#     '''
#     symbolic_opset9.py    
#     def upsample_bilinear2d(g, input, output_size, *args):
#         scales, align_corners = sym_help._get_interpolate_attributes(g, "linear", args)
#         sym_help._interpolate_warning("linear")
#         align_corners = sym_help._maybe_get_scalar(align_corners)    
#         if align_corners:
#             align_corners_ = True
#         else:
#             align_corners_ = False
#         if scales is None:
#             scales = sym_help._interpolate_size_to_scales(g, input, output_size, 4)

#     return g.op("BilinearInterpolate", input, scales, mode_s="linear", align_corners_i=align_corners_)    
#     '''
#     from custom_models.scnn import SCNN
#     net = SCNN(input_size=[512, 288], pretrained=False)

#     runner = Runner("scnn", net, shape, opset_version, fuse)
#     runner.pyotrch_inference()
#     runner.convert()
#     runner.caffe_inference()
#     runner.check_result()

# def test_segnet(shape = [1, 3, 360, 480], opset_version=13, fuse=FUSE):
#     '''
#     symbolic_opset13.py  
#     def max_unpool2d(g, self, indices, output_size):
#         return g.op("MaxUnpool", self, indices, output_size)
#     '''
#     from custom_models.segnet import SegNet
#     net = SegNet()
#     runner = Runner("segnet", net, shape, opset_version, fuse)
#     runner.pyotrch_inference()
#     runner.convert()
#     runner.caffe_inference()
#     runner.check_result()

def test_realcugan(shape = [1, 3, 200, 200], opset_version=13, fuse=FUSE):
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
    pytest.main(['-p', 'no:warnings', '-v', 'test/test_nets.py'])
