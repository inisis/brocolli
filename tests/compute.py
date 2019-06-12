import numpy as np
import argparse


parser = argparse.ArgumentParser(description='test converter')
parser.add_argument('first_file', default=None, metavar='infile_PATH', type=str,
                    help="Path to the first file")
parser.add_argument('second_file', default=None, metavar='infile_PATH', type=str,
                    help="Path to the second file")
args = parser.parse_args()



def Compute(arr1,arr2):
    mult=np.dot(arr1,arr2)
    norm1=np.linalg.norm(arr1)
    norm2=np.linalg.norm(arr2)
    return mult/(norm1*norm2)

if __name__ == '__main__':
    arr1=np.loadtxt(args.first_file,dtype=float)
    arr2=np.loadtxt(args.second_file,dtype=float)
    
    similarity=Compute(arr1,arr2)
    print('shapes are:\n',arr1.shape,'\n',arr2.shape)
    print('Thes similarity is:\n',similarity)
