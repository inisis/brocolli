import os
import pytest
import warnings
import argparse

import torchvision.models as models

from bin.converter.pytorch2onnx import Runner

FUSE = True

os.makedirs("tmp", exist_ok=True)


def test_alexnet(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    net = models.alexnet(pretrained=False)
    runner = Runner("alexnet", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.onnx_inference()
    runner.check_result()


def test_resnet18(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    net = models.resnet18(pretrained=False)
    runner = Runner("resnet18", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.onnx_inference()
    runner.check_result()


def test_squeezenet(shape=(1, 3, 227, 227), opset_version=9, fuse=FUSE):
    net = models.squeezenet1_0(pretrained=False)
    runner = Runner("squeezenet", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.onnx_inference()
    runner.check_result()


def test_googlenet(shape=(1, 3, 224, 224), opset_version=13, fuse=FUSE):
    net = models.googlenet(pretrained=False)
    runner = Runner("googlenet", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.onnx_inference()
    runner.check_result()


def test_mobilenet_v2(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    net = models.mobilenet_v2(pretrained=False)
    runner = Runner("mobilenet", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.onnx_inference()
    runner.check_result()


def test_mobilenet_v3(shape=(1, 3, 224, 224), opset_version=13, fuse=FUSE):
    net = models.mobilenet_v3_small(pretrained=False)
    runner = Runner("mobilenet_v3", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.onnx_inference()
    runner.check_result()


def test_densenet121(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    net = models.densenet121(pretrained=False)
    runner = Runner("densenet121", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.onnx_inference()
    runner.check_result()


def test_densenet161(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    net = models.densenet161(pretrained=False)
    runner = Runner("densenet161", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.onnx_inference()
    runner.check_result()


def test_shufflenet(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    net = models.shufflenet_v2_x1_0(pretrained=False)
    runner = Runner("shufflenet_v2_x1_0", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.onnx_inference()
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
