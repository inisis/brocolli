class CaffeBaseTester(object):
    def __init__(self, name, model, shape, opset_version):
        super(CaffeBaseTester, self).__init__()
        from bin.fx.pytorch2caffe import Runner as CaffeRunner
        self.runner = CaffeRunner(name, model, shape, opset_version)
        self.runner.pyotrch_inference()
        self.runner.convert()
        self.runner.caffe_inference()
        self.runner.check_result()