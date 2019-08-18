# brocolli

a pytorch to caffe && tensorrt model converter, it learns from [MMdnn](https://github.com/Microsoft/MMdnn) && [PytorchConverter](https://github.com/starimeL/PytorchConverter). MMdnn only provides tools for pytorch to onnx conversion, but our tool provides direct conversion from pytorch to caffe && tensorrt

# How to
## run local
> * git clone https://github.com/inisis/caffe (only for valiation if u don't need verify your results, skip this)
> * cd caffe && (revise your dependency to caffe 3) && make pycaffe
> * pip3 install torch==0.4.0 torchvision==0.2.0
> * export PYTHONPATH=$PYTHONPATH:/your/path/to/brocolli/
> * put your pytorch model in tests/pytorch_model folder
> * python3 pytorch_model_converter.py
> * python3 ssd_layer.py example.json pytorch_model/best.pth.prototxt new.prototxt
> * python3 caffe_test.py
> * python3 pytorch_test.py


## run container
> * docker pull yaphets4desmond/pytorch_converter_stable

# Notice 

Curently supported layers
> * Conv
> * PRelu
> * MaxPooling
> * Sigmoid
> * BatchNormalization
> * Relu
> * Add
> * AvgPool
> * Flatten
> * FullyConnected
> * Dropout
> * Softmax
> * Upsample
> * Permute
> * Concat

Curently supported network
> * [SSD](https://github.com/inisis/ssd.pytorch) [Pretrained Weights](https://pan.baidu.com/s/1SqAt-BldJSffZR_1tuQmIw) 
提取码：kiaf
> * ResNet
> * [MTCNN](https://github.com/inisis/DFace) [Pretrained Weights](https://github.com/inisis/DFace/tree/master/model_store) 

# Contact
- Desmond desmond.yao@buaa.edu.cn
- qq group: 597059928
