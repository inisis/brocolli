import pytest
import warnings
import argparse


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Pytorch 2 Onnx operator test.")
    parser.add_argument("--cov", help="foo help")
    args = parser.parse_args()
    if args.cov == "--cov":
        cov = ["--cov", "--cov-report=html:tmp/onnx_report"]
    else:
        cov = []

    pytest.main(["-p", "no:warnings", "-v", "test/converter/op_test/onnx"] + cov)
