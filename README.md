# brocolli

a pytorch to caffe model converter, it learns from [MMdnn](https://github.com/Microsoft/MMdnn) && [PytorchConverter](https://github.com/starimeL/PytorchConverter). MMdnn only provides tools for pytorch to onnx conversion, but our tool provides direct conversion from pytorch to caffe

# How to
## run local
> * pip3 install torch==0.4.0 torchvision==0.2.0
> * export PYTHONPATH=$PYTHONPATH:/home/desmond/Github/brocolli/
> * python3 pytorch_model_converter.py

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

Curently supported network
> * SSD
> * ResNet

# Contact
- Desmond desmond.yao@buaa.edu.cn
- qq group: 597059928
