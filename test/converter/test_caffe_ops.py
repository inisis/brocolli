import os
import pytest
import warnings

os.makedirs("tmp", exist_ok=True)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(["-p", "no:warnings", "-v", "test/converter/op_test/caffe"])
