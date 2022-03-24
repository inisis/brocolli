import os
import re
import sys

import torch
torch.manual_seed(0)
import numpy as np
np.random.seed(0)

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from converter.pytorch.pytorch_trt_parser import PytorchTensorRTParser  # noqa


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class Runner(object):
    def __init__(self, name, model, shape, opset_version):
        self.name = name
        self.model = model
        self.shape = shape
        self.opset_version = opset_version
        self.fuse = True

    def pyotrch_inference(self, generate_onnx=False):
        self.model_file = "tmp/" + self.name
        self.device = torch.device('cpu')
        self.model = self.model.eval().to(self.device)
        if isinstance(self.shape, tuple):
            self.dummy_input = []
            for each in self.shape:
                dummy = torch.rand(each).to(torch.float32)
                self.dummy_input.append(dummy)
        else:
            self.dummy_input = [torch.rand(self.shape).to(torch.float32)]

        self.pytorch_output  = self.model(*self.dummy_input)

        if isinstance(self.pytorch_output , torch.Tensor):
            self.pytorch_output = [self.pytorch_output]    
 
        if generate_onnx:
            torch.onnx.export(self.model, *self.dummy_input, self.name + ".onnx", opset_version=self.opset_version, enable_onnx_checker=False)
        
    def convert(self, export_mode=False):
        self.model.export_mode = export_mode
        pytorch_parser = PytorchTensorRTParser(self.model, self.shape, self.opset_version, self.fuse)
        pytorch_parser.run(self.model_file)

    def trt_inference(self):
        engine_file = "tmp/" + self.name + '.trt'
        self.logger = trt.Logger(trt.Logger.WARNING)

        if isinstance(self.shape, tuple):
            dummy_input = []
            for idx, _ in enumerate(self.shape):
                dummy_input.append(self.dummy_input[idx].numpy())
        else:
            dummy_input = self.dummy_input[0].numpy()

        with trt.Runtime(self.logger) as trt_runtime:
            trt.init_libnvinfer_plugins(None, "")             
            with open(engine_file, 'rb') as f:
                engine_data = f.read()
            engine = trt_runtime.deserialize_cuda_engine(engine_data)
            inputs, outputs, bindings, stream = self.allocate_buffers(engine)

            with engine.create_execution_context() as context:
                if len(inputs) == 1:
                    np.copyto(inputs[0].host, dummy_input.ravel())
                else:   
                    for idx in len(inputs):
                        np.copyto(inputs[idx].host, dummy_input[idx].ravel())        

                for inp in inputs:
                    cuda.memcpy_htod_async(inp.device, inp.host, stream)
                context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)
                for out in outputs:
                    cuda.memcpy_dtoh_async(out.host, out.device, stream) 
                    
                stream.synchronize()

                self.trt_output = [out.host for out in outputs]

    def check_result(self):
        assert len(self.pytorch_output) == len(self.trt_output), "pytorch_output: %d vs trt_output %d" % (len(self.pytorch_output), len(self.trt_output))
        
        for idx in range(len(self.trt_output)):
            np.testing.assert_allclose(
                self.trt_output[idx].flatten(),
                self.pytorch_output[idx].detach().numpy().flatten(),
                rtol=1e-7,
                atol=1e-3, # inception will produce large outputs, but low relative error
            )
        print("accuracy test passed")

    def allocate_buffers(self, engine):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for idx, binding in enumerate(engine):
            size = trt.volume(engine.get_binding_shape(binding)) * 1
            host_mem = cuda.pagelocked_empty(size, dtype=trt.nptype(engine.get_binding_dtype(idx)))
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
