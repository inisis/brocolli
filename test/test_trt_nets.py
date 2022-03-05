import os
os.environ[
    "GLOG_minloglevel"
] = "3"  # 0 - debug 1 - info (still a LOT of outputs) 2 - warnings 3 - errors

import pytest
import warnings

import torchvision.models as models

from bin.pytorch2trt import Runner

os.makedirs('tmp', exist_ok=True)


def test_alexnet(shape = [1, 3, 224, 224], opset_version=9):
    net = models.alexnet(pretrained=False)
    runner = Runner("alexnet", net, shape, opset_version)
    runner.pyotrch_inference()
    runner.convert()
    runner.trt_inference()
    runner.check_result()

def test_resnet18(shape = [1, 3, 224, 224], opset_version=9):
    net = models.resnet18(pretrained=False)
    runner = Runner("resnet18", net, shape, opset_version)
    runner.pyotrch_inference()
    runner.convert()
    runner.trt_inference()
    runner.check_result()

def test_squeezenet(shape = [1, 3, 227, 227], opset_version=9):
    net = models.squeezenet1_0(pretrained=False)
    runner = Runner("squeezenet", net, shape, opset_version)
    runner.pyotrch_inference()
    runner.convert()
    runner.trt_inference()
    runner.check_result()

def test_googlenet(shape = [1, 3, 224, 224], opset_version=9):
    net = models.googlenet(pretrained=False)
    runner = Runner("googlenet", net, shape, opset_version)
    runner.pyotrch_inference()
    runner.convert()
    runner.trt_inference()
    runner.check_result()

def test_yolov5(shape = [1, 3, 640, 640], opset_version=13):
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

    runner = Runner("yolov5", net, shape, opset_version)
    runner.pyotrch_inference(generate_onnx=True)
    runner.convert()
    runner.trt_inference()
    runner.check_result()

def test_resa(shape = [1, 3, 288, 800], opset_version=13):
    from custom_models.resa.configs.culane import cfg
    from custom_models.resa.models.resa import RESANet
    from easydict import EasyDict as edict

    net = RESANet(edict(cfg))

    runner = Runner("resa", net, shape, opset_version)
    runner.pyotrch_inference()
    runner.convert()
    runner.trt_inference()
    runner.check_result()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pytest.main(['-p', 'no:warnings', '-v', 'test/test_trt_nets.py'])
