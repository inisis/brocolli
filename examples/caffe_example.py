import torch
import torchvision.models as models

from brocolli.converter.pytorch_caffe_parser import PytorchCaffeParser

model = models.resnet18(pretrained=False)
x = torch.rand(1, 3, 224, 224)
runner = PytorchCaffeParser(model, x)
runner.convert()
runner.save("resnet18")
runner.check_result()
