class CaffeBaseTester(object):
    def __init__(self, name, model, shape, opset_version):
        super(CaffeBaseTester, self).__init__()
        from bin.converter.pytorch2caffe import Runner as CaffeRunner
        self.runner = CaffeRunner(name, model, shape, opset_version)
        self.runner.pyotrch_inference()
        self.runner.convert()
        self.runner.caffe_inference()
        self.runner.check_result()

class OnnxBaseTester(object):
    def __init__(self, name, model, shape, opset_version):
        super(OnnxBaseTester, self).__init__()
        from bin.converter.pytorch2onnx import Runner as OnnxRunner
        self.runner = OnnxRunner(name, model, shape, opset_version)
        self.runner.pyotrch_inference()
        self.runner.convert()
        self.runner.onnx_inference()
        self.runner.check_result()
