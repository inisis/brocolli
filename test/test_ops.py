import pytest
import argparse
import warnings

parser = argparse.ArgumentParser(description='Brocolli operator test.')
parser.add_argument('frame', type=str, help='caffe, trt or all')

args = parser.parse_args()

def main():
    if args.frame == 'caffe':
        warnings.filterwarnings('ignore')
        pytest.main(['-p', 'no:warnings', '-v', 'test/op_test/caffe'])
    elif args.frame == 'trt':
        warnings.filterwarnings('ignore')
        pytest.main(['-p', 'no:warnings', '-v', 'test/op_test/trt'])
    elif args.frame == 'all': 
        warnings.filterwarnings('ignore')
        pytest.main(['-p', 'no:warnings', '-v', 'test/op_test/'])
    else:
        raise Exception('frame not supported: {}'.format(args.frame))

if __name__ == '__main__':
    main()