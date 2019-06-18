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
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        # forward pass
        y = net(x)  
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        target_num = 0
        list = []
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame,
                            (int(pt[0]), int(pt[1])),
                            (int(pt[2]), int(pt[3])),
                            COLORS[i % 3], 2)
                target_num += 1
                list.append(int(pt[0]))
                list.append(int(pt[1]))
                list.append(int(pt[2]))
                list.append(int(pt[3]))
                j += 1
            
        return frame,target_num,list
    frame = cv2.imread("cat.bmp")
    img,target_num,list_out = predict(frame)
    print(list_out[0], list_out[1], list_out[2], list_out[3])
    count = target_num
    if(target_num > 0):
        full_img_path = 'result.jpg'
        cv2.imwrite(full_img_path, frame)

if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from ssd import build_ssd

    net = build_ssd('test', 300, 21)    # initialize SSD
    net.load_state_dict(torch.load(args.weights))
    transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

    cv2_demo(net.eval(), transform)
   
