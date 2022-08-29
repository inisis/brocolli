class CaffeBaseTester(object):
    def __init__(self, name, model, shape, opset_version):
        super(CaffeBaseTester, self).__init__()
        from brocolli.converter.pytorch_caffe_parser import PytorchCaffeParser

        runner = PytorchCaffeParser(model, shape, opset_version)
        runner.convert()
        runner.save('tmp/' + name)
        runner.check_result()


class OnnxBaseTester(object):
    def __init__(self, name, model, shape, opset_version):
        super(OnnxBaseTester, self).__init__()
        from brocolli.converter.pytorch_onnx_parser import PytorchOnnxParser

        runner = PytorchOnnxParser(model, shape, opset_version)
        runner.convert()
        runner.save('tmp/' + name)
        runner.check_result()
