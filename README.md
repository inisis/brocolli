# brocolli

torch fx based pytorch model converter, including pytorch2caffe  

Pytorch version 1.9.0 and above are all supported  

# How to use
Demo Case
```
git clone https://github.com/inisis/brocolli.git
cd brocolli
python test/fx/test_caffe_nets.py
```

## How to convert your own model
user can follow this sample to convert your own model,
```
from bin.fx.pytorch2caffe import Runner
model = torchvision.models.resnet18(pretrained=False) # Here, you should use your ownd model
runner = Runner("resnet18", model, [1, 3, 224, 224], 13)
# "resnet18": is your converted model name, you should change to your own;
# model: is your own pytorch model, it should be torch.nn.Module
# [1, 3, 224, 224]: is the input shape of your model
runner.pyotrch_inference()
runner.convert()
runner.caffe_inference()
runner.check_result()
```
user can run this script until you see "accuracy test passed" on screen, then you can get your caffe or trt model under tmp folder.

# Notice 

torch fx is highly recommended, and torch jit is deprecated, please do not use torch jit to convert your model, it will be removed soon.  

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
 
 ![image](imgs/QGRPOUP.png)
