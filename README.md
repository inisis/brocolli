# brocolli

a pytorch to caffe && tensorrt model converter, our tool provides direct conversion from pytorch to caffe && tensorrt.

Support 1.9.0 or higher Pytorch

# How to use
> * clone caffe from https://github.com/inisis/caffe and build
> * export PYTHONPATH=/path/to/your/caffe/python:$PYTHONPATH
> * python test/test_nets.py

# Notice 

Curently supported layers
> * Conv
> * PRelu
> * MaxPooling
> * Sigmoid
> * BatchNormalization
> * Relu
> * LeakyRelu
> * Add
> * AvgPool
> * Flatten
> * FullyConnected
> * Dropout
> * Softmax
> * Upsample
> * Permute
> * Concat
> * Unsqueeze
> * Relu6
> * Pad
> * HardSwish
> * HardSigmoid
> * Mul    
> * Slice 
> * L2Normalization
> * Resize
> * ReduceMean
> * BilinearInterpolate
> * MaxUnPool
> * ConvTranspose

Curently supported network
> * SSD
> * AlexNet
> * ResNet
> * GoogleNet
> * SqueezeNet
> * MobileNet
> * DenseNet
> * Inception
> * VGG
> * YoloV3
> * ShuffleNet
> * SCNN
> * SegNet
> * YoloV5


# Contact
 QQ Group: 597059928
 
 ![image](imgs/QGRPOUP.png)