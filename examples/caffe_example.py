import torchvision.models as models

from brocolli.converter.pytorch_caffe_parser import PytorchCaffeParser

model = models.resnet18(pretrained=False)
runner = PytorchCaffeParser(model, (1, 3, 224, 224))
runner.convert()
runner.save("resnet18")
runner.check_result()
