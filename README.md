# brocolli

torch fx based pytorch model converter, including pytorch2caffe, pytorch2onnx.  
torch fx based pytorch model quantizier.

# Installation
```
pip install brocolli
```

# How to use
* torch2caffe
    * caffe installation
    ```bash
    pip install brocolli-caffe
    ```

    ```
    import torchvision.models as models
    from brocolli.converter.pytorch_caffe_parser import PytorchCaffeParser

    net = models.alexnet(pretrained=False)
    x = torch.rand(1, 3, 224, 224)
    pytorch_parser = PytorchCaffeParser(net, x)
    pytorch_parser.convert()
    pytorch_parser.save('alexnet')
    ```
    run this script until you see "accuracy test passed" on screen, then you can get alexnet.caffemodel and alexnet.prototxt under under current folder.

* torch2onnx
    ```
    import torchvision.models as models
    from brocolli.converter.pytorch_onnx_parser import PytorchOnnxParser

    net = models.alexnet(pretrained=False)
    x = torch.rand(1, 3, 224, 224)
    pytorch_parser = PytorchCaffeParser(net, x)
    pytorch_parser.convert()
    pytorch_parser.save('alexnet.onnx')
    ```
    run this script until you see "accuracy test passed" on screen, then you can get alexnet.onnx under current folder.

# Contact
 QQ Group: 597059928
 
 ![image](https://raw.githubusercontent.com/inisis/brocolli/master/imgs/QGRPOUP.png)
