class CaffeBaseTester(object):
    def __init__(self, name, model, shape, fuse=False):
        super(CaffeBaseTester, self).__init__()
        from brocolli.converter.pytorch_caffe_parser import PytorchCaffeParser

        runner = PytorchCaffeParser(model, shape, fuse)
        runner.convert()
        runner.save("tmp/" + name)
        runner.check_result()


class OnnxBaseTester(object):
    def __init__(
        self, name, model, inputs, fuse=False, concrete_args=None, dynamic_batch=False
    ):
        super(OnnxBaseTester, self).__init__()
        from brocolli.converter.pytorch_onnx_parser import PytorchOnnxParser

        runner = PytorchOnnxParser(model, inputs, fuse, concrete_args, dynamic_batch)
        runner.convert()
        runner.save("tmp/" + name + ".onnx")
        runner.check_result()
