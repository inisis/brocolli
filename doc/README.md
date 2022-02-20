# Development doc

## Caffe

### layer with multi output
1. layers like Slice and max_pool_with_indice can have more than one output, so each output should be handled properly.

## Onnx

### Shape inference
1. shape can be retrived from torch.C._Node, but not every node has a shape;

### Input to attribute
1. For different version of onnx, attribute can be moved to input, and input can be moved to attribute, but attribute is static, input is dynamic.
