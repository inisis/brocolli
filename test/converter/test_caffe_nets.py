import os

os.environ[
    "GLOG_minloglevel"
] = "3"  # 0 - debug 1 - info (still a LOT of outputs) 2 - warnings 3 - errors

import pytest
import warnings

import torchvision.models as models

from brocolli.converter.pytorch_caffe_parser import PytorchCaffeParser

FUSE = True

os.makedirs("tmp", exist_ok=True)


def test_alexnet(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    model = models.alexnet(pretrained=False)
    runner = PytorchCaffeParser(model, shape, opset_version)
    runner.convert()
    runner.save('tmp/alexnet')
    runner.check_result()


def test_resnet18(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    model = models.resnet18(pretrained=False)
    runner = PytorchCaffeParser(model, shape, opset_version)
    runner.convert()
    runner.save('tmp/resnet18')
    runner.check_result()


def test_squeezenet(shape=(1, 3, 227, 227), opset_version=9, fuse=FUSE):
    model = models.squeezenet1_0(pretrained=False)
    runner = PytorchCaffeParser(model, shape, opset_version)
    runner.convert()
    runner.save('tmp/squeezenet')
    runner.check_result()


def test_googlenet(shape=(1, 3, 224, 224), opset_version=13, fuse=FUSE):
    model = models.googlenet(pretrained=False)
    runner = PytorchCaffeParser(model, shape, opset_version)
    runner.convert()
    runner.save('tmp/googlenet')
    runner.check_result()


def test_mobilenet_v2(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    model = models.mobilenet_v2(pretrained=False)
    runner = PytorchCaffeParser(model, shape, opset_version)
    runner.convert()
    runner.save('tmp/mobilenet')
    runner.check_result()


def test_mobilenet_v3(shape=(1, 3, 224, 224), opset_version=13, fuse=FUSE):
    model = models.mobilenet_v3_small(pretrained=False)
    runner = PytorchCaffeParser(model, shape, opset_version)
    runner.convert()
    runner.save('tmp/mobilenet_v3')
    runner.check_result()


def test_densenet121(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    model = models.densenet121(pretrained=False)
    runner = PytorchCaffeParser(model, shape, opset_version)
    runner.convert()
    runner.save('tmp/densenet121')
    runner.check_result()


def test_densenet161(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    model = models.densenet161(pretrained=False)
    runner = PytorchCaffeParser(model, shape, opset_version)
    runner.convert()
    runner.save('tmp/densenet161')
    runner.check_result()


def test_shufflenet(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    model = models.shufflenet_v2_x1_0(pretrained=False)
    runner = PytorchCaffeParser(model, shape, opset_version)
    runner.convert()
    runner.save('tmp/shufflenet')
    runner.check_result()


def test_ssd300_vgg16(shape=(1, 3, 300, 300), opset_version=13, fuse=FUSE):
    from custom_models.ssd import build_ssd

    model = build_ssd("export")
    runner = PytorchCaffeParser(model, shape, opset_version)
    runner.convert()
    runner.save('tmp/ssd300_vgg16')
    runner.check_result()


def test_yolov5(shape=(1, 3, 640, 640), opset_version=13, fuse=FUSE):
    import torch
    concrete_args = {"augment": False, "profile": False, "visualize": False}
    model = torch.hub.load(
        "ultralytics/yolov5",
        "yolov5s",
        autoshape=False,
        pretrained=False,
        device=torch.device("cpu"),
    )

    class Identity(torch.nn.Module):
        def __init__(self):
            super(Identity, self).__init__()

        def forward(self, x):
            for i in range(self.nl):
                x[i] = self.m[i](x[i])
                bs, _, ny, nx = x[i].shape
                x[i] = (
                    x[i]
                    .view(bs, self.na, self.no, ny, nx)
                    .permute(0, 1, 3, 4, 2)
                    .contiguous()
                )

            return x

    name, _ = list(model.model.named_children())[-1]
    identity = Identity()
    detect = getattr(model.model, name)
    identity.__dict__.update(detect.__dict__)
    setattr(model.model, name, identity)

    runner = PytorchCaffeParser(model, shape, opset_version, concrete_args=concrete_args)
    runner.convert()
    runner.save('tmp/yolov5')
    runner.check_result()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(["-p", "no:warnings", "-v", "test/converter/test_caffe_nets.py"])
