from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
import argparse
from data import BaseTransform 

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='./pytorch_model/ssd_300_VOC.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda in live demo')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

def cv2_demo(net, transform):
    def predict(frame):
        height, width = frame.shape[:2]
        # np.ndarray -> torch.Tensor.
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1).numpy()
        # forward pass
        net.blobs['data'].data[...] = x
        detections = net.forward()['detection_out']
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        target_num = 0
        det_label = detections[0,0,:,1]# 标签索引
        det_conf = detections[0,0,:,2] # 可信度
        det_xmin = detections[0,0,:,3] # 坐标
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.5]
        top_conf = det_conf[top_indices]# 可信度
        top_label_indices = det_label[top_indices].tolist()# 标签索引
        top_labels = "unknown"
        top_xmin = det_xmin[top_indices]# 坐标 0~1小数
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in range(min(5, top_conf.shape[0])):# 前5个
            xmin = top_xmin[i] * width # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i] * height# ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i] * width# xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i] * height# ymax = int(round(top_ymax[i] * image.shape[0]))
            print(xmin, ymin, xmax, ymax)
            cv2.rectangle(frame,
                            (int(xmin), int(ymin)),
                            (int(xmax), int(ymax)),
                            COLORS[i % 3], 2)
            score = top_conf[i] 
            label = int(top_label_indices[i])#标签id
            label_name = top_labels[i]#标签字符串
            result.append([xmin, ymin, xmax, ymax, label, score, label_name])
            
        return frame, len(result), result

    frame = cv2.imread("cat.bmp")
    img,target_num,list_out = predict(frame)
    print(target_num)
    count = target_num
    if(target_num > 0):
        full_img_path = 'result.jpg'
        cv2.imwrite(full_img_path, frame)

if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    sys.path.append('/tool/caffe/python')
    sys.path.append('/tool/caffe/python/caffe')
    import caffe

    model_file = "pytorch_model/ssd_300_VOC"
    Model_FILE = model_file + '.prototxt'

    PRETRAINED = model_file + '.caffemodel'
    net = caffe.Classifier(Model_FILE, PRETRAINED)

    transform = BaseTransform(300, (104/256.0, 117/256.0, 123/256.0))

    cv2_demo(net, transform)
   
