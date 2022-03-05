# brocolli

a pytorch to caffe && tensorrt model converter, our tool provides direct conversion from pytorch to caffe && tensorrt.

Support 1.9.0 or higher Pytorch

# How to use
docker image is provided, you can use following command to get a stable development env.

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
|LeakyRelu           |✔️|❔|
|Add                 |✔️|✔️|
|AvgPool             |✔️|✔️|
|Flatten             |✔️|✔️|
|FullyConnected      |✔️|✔️|
|Softmax             |✔️|✔️|
|Upsample            |✔️|✔️|
|Permute             |✔️|✔️|
|Concat              |✔️|✔️|
|Unsqueeze           |✔️|❔|
|Relu6               |✔️|❔|
|Pad                 |✔️|❔|
|HardSwish           |✔️|❔|
|HardSigmoid         |✔️|❔|
|Mul                 |✔️|✔️|
|Slice               |✔️|❔|
|L2Normalization     |✔️|❔|
|Resize              |✔️|✔️|
|ReduceMean          |✔️|❔|
|BilinearInterpolate |✔️|✔️|
|MaxUnPool           |✔️|❔|
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
|MobileNet |✔️|❔|
|DenseNet  |✔️|❔|
|VGG       |✔️|❔|
|SCNN      |✔️|❔|
|SegNet    |✔️|❔|
|YoloV5    |✔️|✔️|
|Realcugan |✔️|❔|
|Yolo-Lite |✔️|❔|
|Resa      |❌|✔️|


# Contact
 QQ Group: 597059928
 
 ![image](imgs/QGRPOUP.png)