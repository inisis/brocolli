import torch.nn as nn
import torch.nn.functional as F
import pytest
import warnings

from brocolli.testing.common_utils import OnnxBaseTester as Tester

# def test_BatchNorm2d(shape = [1, 3, 32, 32], opset_version=13):
#     model = torch.nn.BatchNorm2d(3)
#     Tester("BatchNorm2d", model, shape, opset_version)

class L2Norm(nn.Module):
    def __init__(self,):
        super(L2Norm, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=1)

def test_L2Norm(shape = (1, 3, 32, 32), opset_version=13):
    model = L2Norm()
    # dummy_input = torch.rand(shape).to(torch.float32)
    # torch.onnx.export(model, (dummy_input,), "l2norm.onnx", opset_version=13, enable_onnx_checker=False)
    Tester("L2Norm", model, shape, opset_version)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pytest.main(['-p', 'no:warnings', '-v', 'test/converter/op_test/onnx/normalization/test_normalization.py'])
