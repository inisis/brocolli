# import os
# import sys
# import torch
# import pytest
# import warnings

# from bin.converter.utils import CaffeBaseTester as Tester

# def test_BatchNorm2d(shape = [1, 3, 32, 32],):
#     model = torch.nn.BatchNorm2d(3)
#     Tester("BatchNorm2d", model, shape, opset_version)


# if __name__ == '__main__':
#     warnings.filterwarnings('ignore')
#     pytest.main(['-p', 'no:warnings', '-v', 'test/op_test/caffe/normalization/test_normalization.py'])
