import torch.nn as nn
import torch.nn.functional as F
import pytest
import warnings

from brocolli.testing.common_utils import OnnxBaseTester as Tester


class L2Norm(nn.Module):
    def __init__(self,):
        super(L2Norm, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=1)

def test_L2Norm(shape = (1, 3, 32, 32), opset_version=13):
    model = L2Norm()
    Tester("L2Norm", model, shape, opset_version)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pytest.main(['-p', 'no:warnings', '-v', 'test/converter/op_test/onnx/normalization/test_normalization.py'])
