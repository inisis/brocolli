import os
import sys
import torch
import pytest
import warnings
import argparse

import torchvision.models as models

from brocolli.converter.pytorch_onnx_parser import PytorchOnnxParser

FUSE = True
PRETRAINED = False

os.makedirs("tmp", exist_ok=True)


def test_alexnet(shape=(1, 3, 224, 224), fuse=FUSE):
    model = models.alexnet(pretrained=PRETRAINED)
    x = torch.rand(shape)
    runner = PytorchOnnxParser(model, x, fuse)
    runner.convert()
    runner.save("tmp/alexnet.onnx")
    runner.check_result()


def test_resnet18(shape=(1, 3, 224, 224), fuse=FUSE):
    model = models.resnet18(pretrained=PRETRAINED)
    x = torch.rand(shape)
    runner = PytorchOnnxParser(model, x, fuse, dynamic_batch=True)
    runner.convert()
    runner.save("tmp/resnet18.onnx")
    runner.check_result()


def test_squeezenet(shape=(1, 3, 227, 227), fuse=FUSE):
    model = models.squeezenet1_0(pretrained=PRETRAINED)
    x = torch.rand(shape)
    runner = PytorchOnnxParser(model, x, fuse)
    runner.convert()
    runner.save("tmp/squeezenet.onnx")
    runner.check_result()


def test_googlenet(shape=(1, 3, 224, 224), fuse=FUSE):
    model = models.googlenet(pretrained=PRETRAINED)
    x = torch.rand(shape)
    runner = PytorchOnnxParser(model, x, fuse)
    runner.convert()
    runner.save("tmp/googlenet.onnx")
    runner.check_result()


def test_mobilenet_v2(shape=(1, 3, 224, 224), fuse=FUSE):
    model = models.mobilenet_v2(pretrained=PRETRAINED)
    x = torch.rand(shape)
    runner = PytorchOnnxParser(model, x, fuse)
    runner.convert()
    runner.save("tmp/mobilenet.onnx")
    runner.check_result()


def test_mobilenet_v3_small(shape=(128, 3, 224, 224), fuse=FUSE):
    model = models.mobilenet_v3_small(pretrained=PRETRAINED)
    x = torch.rand(shape)
    runner = PytorchOnnxParser(model, x, fuse)
    runner.convert()
    runner.save("tmp/mobilenet_v3_small.onnx")
    runner.check_result()


def test_mobilenet_v3_large(shape=(128, 3, 224, 224), fuse=FUSE):
    model = models.mobilenet_v3_large(pretrained=PRETRAINED)
    x = torch.rand(shape)
    runner = PytorchOnnxParser(model, x, fuse)
    runner.convert()
    runner.save("tmp/mobilenet_v3_large.onnx")
    runner.check_result()


def test_densenet121(shape=(1, 3, 224, 224), fuse=FUSE):
    model = models.densenet121(pretrained=PRETRAINED)
    x = torch.rand(shape)
    runner = PytorchOnnxParser(model, x, fuse)
    runner.convert()
    runner.save("tmp/densenet121.onnx")
    runner.check_result()


def test_densenet161(shape=(1, 3, 224, 224), fuse=FUSE):
    model = models.densenet161(pretrained=PRETRAINED)
    x = torch.rand(shape)
    runner = PytorchOnnxParser(model, x, fuse)
    runner.convert()
    runner.save("tmp/densenet161.onnx")
    runner.check_result()


def test_shufflenet(shape=(1, 3, 224, 224), fuse=FUSE):
    model = models.shufflenet_v2_x1_0(pretrained=PRETRAINED)
    x = torch.rand(shape)
    runner = PytorchOnnxParser(model, x, fuse)
    runner.convert()
    runner.save("tmp/shufflenet.onnx")
    runner.check_result()


@pytest.mark.skipif("cnocr" not in sys.modules, reason="requires the cnocr library")
def test_cnocr(shape=(1, 3, 224, 224), fuse=FUSE):
    from cnocr import CnOcr

    ocr = CnOcr(rec_model_name="densenet_lite_136-gru", rec_model_backend="pytorch")
    model = ocr.rec_model._model
    img = torch.randn(1, 1, 32, 1024)
    y = torch.rand(1)
    concrete_args = {
        "target": None,
        "candidates": None,
        "return_logits": True,
        "return_preds": False,
    }
    runner = PytorchOnnxParser(model, (img, y), concrete_args=concrete_args)
    runner.convert()
    runner.save("tmp/cnocr.onnx")
    runner.check_result()


def test_yolov5(shape=(1, 3, 640, 640), fuse=FUSE):
    concrete_args = {"augment": False, "profile": False, "visualize": False}
    model = torch.hub.load(
        "ultralytics/yolov5",
        "yolov5s",
        autoshape=False,
        pretrained=True,
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

    name, _ = list(model.model.model.named_children())[-1]

    identity = Identity()
    detect = getattr(model.model.model, name)
    identity.__dict__.update(detect.__dict__)
    setattr(model.model.model, name, identity)
    x = torch.rand(shape)

    runner = PytorchOnnxParser(model.model, x, concrete_args=concrete_args)
    runner.convert()
    runner.save("yolov5_lp.onnx")
    runner.check_result()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Pytorch 2 Onnx network test.")
    parser.add_argument("--cov", help="foo help")
    args = parser.parse_args()
    if args.cov == "--cov":
        cov = ["--cov", "--cov-report=html:tmp/onnx_report"]
    else:
        cov = []

    pytest.main(["-p", "no:warnings", "-v", "test/converter/test_onnx_nets.py"] + cov)
