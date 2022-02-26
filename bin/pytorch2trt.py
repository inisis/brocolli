import os
import re
import sys

import torch
torch.manual_seed(0)
import numpy as np
np.random.seed(0)

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import tensorrt as trt
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
            dummy_input = []
            for each in self.shape:
                dummy = torch.ones(each).to(torch.float32)
                dummy_input.append(dummy)
        else:
            dummy_input = torch.ones(self.shape).to(torch.float32)

        self.pytorch_output = self.model(dummy_input)
 
        if generate_onnx:
            torch.onnx.export(self.model, dummy_input, self.name + ".onnx", opset_version=self.opset_version, enable_onnx_checker=False)
        
    def convert(self, export_mode=False):
        self.model.export_mode = export_mode
        pytorch_parser = PytorchTensorRTParser(self.model, self.shape, self.opset_version, self.fuse)
        pytorch_parser.run(self.model_file)

    def trt_inference(self):
        def load_engine(trt_runtime, engine_path):
            trt.init_libnvinfer_plugins(None, "")             
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            engine = trt_runtime.deserialize_cuda_engine(engine_data)

            return engine

        engine_file = "tmp/" + self.name + '.trt'
        self.dtype = np.float32
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = load_engine(self.runtime, engine_file)
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

        if isinstance(self.shape, tuple):
            dummy_input = []
            for each in self.shape:
                dummy = np.ones(each)
                dummy_input.append(dummy)
        else:
            dummy_input = np.ones(self.shape)

        x = dummy_input
        
        np.copyto(self.inputs[0].host, x.ravel())
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        self.context.execute_async(batch_size=1, bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
            
        self.stream.synchronize()
        self.outputs = self.outputs[0].host

        del self.engine
        del self.runtime
        del self.context

    def check_result(self):
        np.testing.assert_allclose(
            self.outputs.flatten(),
            self.pytorch_output.detach().numpy().flatten(),
            rtol=1e-7,
            atol=1e-3, # inception will produce large outputs, but low relative error
        )
        print("accuracy test passed")

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * 1
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
