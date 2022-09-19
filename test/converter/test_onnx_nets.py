import os
import pytest
import warnings
import argparse

import torchvision.models as models

from brocolli.converter.pytorch_onnx_parser import PytorchOnnxParser

FUSE = True

os.makedirs("tmp", exist_ok=True)


def test_alexnet(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    model = models.alexnet(pretrained=False)
    runner = PytorchOnnxParser(model, shape, opset_version)
    runner.convert()
    runner.save("tmp/alexnet.onnx")
    runner.check_result()


def test_resnet18(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    model = models.resnet18(pretrained=False)
    runner = PytorchOnnxParser(model, shape, opset_version)
    runner.convert()
    runner.save("tmp/resnet18.onnx")
    runner.check_result()


def test_squeezenet(shape=(1, 3, 227, 227), opset_version=9, fuse=FUSE):
    model = models.squeezenet1_0(pretrained=False)
    runner = PytorchOnnxParser(model, shape, opset_version)
    runner.convert()
    runner.save("tmp/squeezenet.onnx")
    runner.check_result()


def test_googlenet(shape=(1, 3, 224, 224), opset_version=13, fuse=FUSE):
    model = models.googlenet(pretrained=False)
    runner = PytorchOnnxParser(model, shape, opset_version)
    runner.convert()
    runner.save("tmp/googlenet.onnx")
    runner.check_result()


def test_mobilenet_v2(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    model = models.mobilenet_v2(pretrained=False)
    runner = PytorchOnnxParser(model, shape, opset_version)
    runner.convert()
    runner.save("tmp/mobilenet.onnx")
    runner.check_result()


def test_mobilenet_v3(shape=(1, 3, 224, 224), opset_version=13, fuse=FUSE):
    model = models.mobilenet_v3_small(pretrained=False)
    runner = PytorchOnnxParser(model, shape, opset_version)
    runner.convert()
    runner.save("tmp/mobilenet_v3.onnx")
    runner.check_result()


def test_densenet121(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    model = models.densenet121(pretrained=False)
    runner = PytorchOnnxParser(model, shape, opset_version)
    runner.convert()
    runner.save("tmp/densenet121.onnx")
    runner.check_result()


def test_densenet161(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    model = models.densenet161(pretrained=False)
    runner = PytorchOnnxParser(model, shape, opset_version)
    runner.convert()
    runner.save("tmp/densenet161.onnx")
    runner.check_result()


def test_shufflenet(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    model = models.shufflenet_v2_x1_0(pretrained=False)
    runner = PytorchOnnxParser(model, shape, opset_version)
    runner.convert()
    runner.save("tmp/shufflenet.onnx")
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
