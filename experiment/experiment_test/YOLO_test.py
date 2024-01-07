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


RESULT_PATH = '../results/YOLO_dark_results/'
KITTI_TRAIN_PATH = '/media/server1/5150/Wu/KITTI/training'
VAL_PATH = '../../ImageSets/val.txt'

data_loader_train = Dataset(KITTI_TRAIN_PATH + '/velodyne_reduced', KITTI_TRAIN_PATH + '/calib',
                            KITTI_TRAIN_PATH + '/image_2_dark', KITTI_TRAIN_PATH + '/label_2', val_image_ids=VAL_PATH)

YOLO_model = YOLO("../../yolov8n.pt")
for idx,((point_file, calib_file), img_file, gt_bbox) in enumerate(tqdm(data_loader_train, desc='Data no.')):
    if VAL_PATH:
        idx = data_loader_train.val_image_ids[idx]
    print(idx)
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
            useless1 = '0.00 0 0.00'
            useless2 = ' '.join(['0.00' for i in range(7)])
            content = ' '.join(['Pedestrian',useless1,' '.join(['{:.2f}'.format(i) for i in YOLO_results[i][:-1]]),
                                useless2,'{:.2f}'.format(YOLO_results[i][-1])])
            print(content)
            f.write(content+'\n')
