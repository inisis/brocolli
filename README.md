# brocolli

a pytorch to caffe model converter, it learns from [MMdnn](https://github.com/Microsoft/MMdnn) && [PytorchConverter](https://github.com/starimeL/PytorchConverter). MMdnn only provides tools for pytorch to onnx conversion, but our tool provides direct conversion from pytorch to caffe

# How to

> * pip install torch==0.4.0 torchvision
> * export PYTHONPATH=$PYTHONPATH:/home/desmond/Github/brocolli/
> * python test_pytorch.py

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


# Contact
- Desmond desmond.yao@buaa.edu.cn
