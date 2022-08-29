# brocolli

torch fx based pytorch model converter, including pytorch2caffe, pytorch2onnx.  
torch fx based pytorch model quantizier.

Pytorch version 1.9.0 and above are all supported  

# installation
```
pip install brocolli
```

# How to use
```
import torchvision.models as models
from brocolli.converter.pytorch_caffe_parser import PytorchCaffeParser

net = models.alexnet(pretrained=False)
pytorch_parser = PytorchCaffeParser(net, [(1, 3, 224, 223)])
pytorch_parser.convert()
pytorch_parser.save('alexnet.onnx')
```
user can run this script until you see "accuracy test passed" on screen, then you can get your caffe or trt model under tmp folder.

# Notice 
* ✔️ : support 
* ❔ : shall support
* ❌ : not support

Curently supported layers

|                    |Caffe|TensorRT|
|---                 |---|---|
|Conv                |✔️|✔️|
|PRelu               |✔️|❔|
|MaxPooling          |✔️|✔️|
|Sigmoid             |✔️|✔️|
|BatchNormalization  |✔️|✔️|
|Relu                |✔️|✔️|
|LeakyRelu           |✔️|✔️|
|Add                 |✔️|✔️|
|AvgPool             |✔️|✔️|
|Flatten             |✔️|✔️|
|FullyConnected      |✔️|✔️|
|Softmax             |✔️|✔️|
|Upsample            |✔️|✔️|
|Permute             |✔️|✔️|
|Concat              |✔️|✔️|
|Unsqueeze           |✔️|❔|
|Relu6               |✔️|✔️|
|Pad                 |✔️|✔️|
|HardSwish           |✔️|✔️|
|HardSigmoid         |✔️|✔️|
|Mul                 |✔️|✔️|
|Slice               |✔️|✔️|
|L2Normalization     |✔️|❔|
|Resize              |✔️|✔️|
|ReduceMean          |✔️|✔️|
|BilinearInterpolate |✔️|✔️|
|MaxUnPool           |✔️|❌|
|ConvTranspose       |✔️|✔️|
|Gather              |❌|✔️|
|PixelShufle         |✔️|❔|


Curently supported network

|          |Caffe|TensorRT|
|---       |---|---|
|SSD       |✔️|❔|
|AlexNet   |✔️|✔️|
|ResNet    |✔️|✔️|
|GoogleNet |✔️|✔️|
|SqueezeNet|✔️|✔️|
|MobileNet |✔️|✔️|
|DenseNet  |✔️|✔️|
|ShuffleNet|✔️|✔️|
|SCNN      |✔️|✔️|
|SegNet    |✔️|❌|
|YoloV5    |✔️|✔️|
|YoloV3    |✔️|✔️|
|Realcugan |✔️|❔|
|Yolo-Lite |✔️|❔|
|Resa      |❌|✔️|
|YoloX     |✔️|✔️|
|BiSeNet   |❌|✔️|
|fbnet     |✔️|❔|
|regnet    |✔️|❔|
|ghostnet  |✔️|❔|
|tinynet   |✔️|❔|
|YoloV7    |✔️|❔|

# TODO
RNN support

# Contact
 QQ Group: 597059928
 
 ![image](https://raw.githubusercontent.com/inisis/brocolli/master/imgs/QGRPOUP.png)
