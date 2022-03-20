# brocolli

a pytorch to caffe && tensorrt model converter, our tool provides direct conversion from pytorch to caffe && tensorrt.

Support 1.9.0 or higher Pytorch

# How to use
⚠️user must uses provided docker to convert your model, clone code only currently will not work.

for Caffe-only:
```
docker pull yaphets4desmond/brocolli:v1.0
docker run --rm --name=BRO -it yaphets4desmond/brocolli:v1.0 bash
cd /root/brocolli && python test/test_caffe_nets.py
```
for TensorRT:
```
docker pull yaphets4desmond/brocolli:v2.0
docker run --gpus=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility --rm --name=BRO -it yaphets4desmond/brocolli:v2.0 bash
cd /root/brocolli && python test/test_trt_nets.py
```

the source code is located in /root/brocolli, and a pre-compiled caffe is in /root/caffe

## How to convert your own model
user can follow this sample to convert your own model,
```
from bin.pytorch2caffe import Runner # if caffe, use bin.pytorch2caffe, if TensorRT use bin.pytorch2trt;
model = torchvision.models.resnet18(pretrained=False) # Here, you should use your ownd model
runner = Runner("resnet18", model, [1, 3, 224, 224], 13)
# "resnet18": is your converted model name, you should change to your own;
# model: is your own pytorch model, it should be torch.nn.Module
# [1, 3, 224, 224]: is the input shape of your model
# 13: is the op_set version, use 13 by default
runner.pyotrch_inference()
runner.convert()
runner.caffe_inference() # if caffe, use caffe_inference, if TensorRT use trt_inference;
runner.check_result()
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


# Contact
 QQ Group: 597059928
 
 ![image](imgs/QGRPOUP.png)