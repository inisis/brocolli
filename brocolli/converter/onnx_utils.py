import torch
import numpy as np
from onnx import TensorProto as tp


scalar_type_to_pytorch_type = [
    torch.uint8,  # 0
    torch.int8,  # 1
    torch.short,  # 2
    torch.int,  # 3
    torch.int64,  # 4
    torch.half,  # 5
    torch.float,  # 6
    torch.double,  # 7
    torch.complex32,  # 8
    torch.complex64,  # 9
    torch.complex128,  # 10
    torch.bool,  # 11
]

cast_pytorch_to_onnx = {
    "Byte": tp.UINT8,
    "Char": tp.INT8,
    "Double": tp.DOUBLE,
    "Float": tp.FLOAT,
    "Half": tp.FLOAT16,
    "Int": tp.INT32,
    "Long": tp.INT64,
    "Short": tp.INT16,
    "Bool": tp.BOOL,
    "ComplexFloat": tp.COMPLEX64,
    "ComplexDouble": tp.COMPLEX128,
    "Undefined": tp.UNDEFINED,
}

scalar_type_to_onnx = [
    cast_pytorch_to_onnx["Byte"],
    cast_pytorch_to_onnx["Char"],
    cast_pytorch_to_onnx["Short"],
    cast_pytorch_to_onnx["Int"],
    cast_pytorch_to_onnx["Long"],
    cast_pytorch_to_onnx["Half"],
    cast_pytorch_to_onnx["Float"],
    cast_pytorch_to_onnx["Double"],
    cast_pytorch_to_onnx["Undefined"],
    cast_pytorch_to_onnx["ComplexFloat"],
    cast_pytorch_to_onnx["ComplexDouble"],
    cast_pytorch_to_onnx["Bool"],
]

numpy_to_torch_dtype_dict = {
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}

torch_to_numpy_dtype_dict = {
    value: key for (key, value) in numpy_to_torch_dtype_dict.items()
}


def pytorch_dtype_to_onnx(scalar_type):
    torch_type = scalar_type_to_pytorch_type.index(scalar_type)
    onnx_type = scalar_type_to_onnx[torch_type]
    return onnx_type


def numpy_dtype_to_torch(scalar_type):
    return numpy_to_torch_dtype_dict[scalar_type]


def torch_dtype_to_numpy(scalar_type):
    return torch_to_numpy_dtype_dict[scalar_type]
