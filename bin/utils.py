from bin.pytorch2caffe import Runner as CaffeRunner
from bin.pytorch2trt import Runner as TensorRTRunner

class CaffeBaseTester(object):
    def __init__(self, name, model, shape, opset_version):
        super(CaffeBaseTester, self).__init__()
        self.runner = CaffeRunner(name, model, shape, opset_version)
        self.runner.pyotrch_inference()
        self.runner.convert()
        self.runner.caffe_inference()
        self.runner.check_result()

class TensorRTBaseTester(object):
    def __init__(self, name, model, shape, opset_version):
        super(TensorRTBaseTester, self).__init__()
        self.runner = TensorRTRunner(name, model, shape, opset_version)
        self.runner.pyotrch_inference()
        self.runner.convert()
        self.runner.trt_inference()
        self.runner.check_result()