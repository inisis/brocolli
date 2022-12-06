import torch
import torchvision.models as models

from brocolli.converter.pytorch_onnx_parser import PytorchOnnxParser

model = models.resnet18(pretrained=False)
x = torch.rand((1 ,3 ,224, 224))
runner = PytorchOnnxParser(model, x)
runner.convert()
runner.save("resnet18.onnx")
runner.check_result()
