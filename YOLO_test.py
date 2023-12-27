import numpy
import torch
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import time
from MLP_model import MLPModel
from Dataloader import Dataset
from ultralytics import YOLO
from MLP_utils import *
from tqdm import tqdm

import sys


RESULT_PATH = './YOLO_results/'
KITTI_TRAIN_PATH = '/media/server1/5150/Wu/KITTI/training'

data_loader_train = Dataset(KITTI_TRAIN_PATH + '/velodyne_reduced', KITTI_TRAIN_PATH + '/calib',
                            KITTI_TRAIN_PATH + '/image_2', KITTI_TRAIN_PATH + '/label_2')

YOLO_model = YOLO("yolov8n.pt")
idx = 0
for (point_file, calib_file), img_file, gt_boxes in tqdm(data_loader_train, desc='Data no.'):
    '''
    YOLO predict
    '''
    YOLO_boxes = YOLO_model.predict(img_file, classes=0)[0].boxes
    YOLO_results = []
    for box in YOLO_boxes:
        if box.cls == 0:
            YOLO_results.append(torch.concat([box.xyxy.squeeze(), box.conf], dim=0).tolist())
    YOLO_results = numpy.array(YOLO_results)
    # print('YOLO result:',YOLO_results)
    with open(RESULT_PATH+f'{str(idx).zfill(6)}.txt','w') as f:
        # if len(YOLO_results) == 0:
        #     f.write(' '.join(['DontCare',' '.join(['0' for i in range(15)])]))
        for i in range(len(YOLO_results)):
            useless1 = ' '.join(['0' for i in range(3)])
            useless2 = ' '.join(['0' for i in range(7)])
            content = ' '.join(['Pedestrain',useless1,' '.join([str(i) for i in YOLO_results[i][:-1]]),useless2,str(YOLO_results[i][-1])])
            print(content)
            f.write(content+'\n')
    idx += 1