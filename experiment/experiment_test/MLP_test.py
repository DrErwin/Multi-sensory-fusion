import numpy as np
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
import sys

from tqdm import tqdm

sys.path.append('../..')
sys.path.append('../../PointPillars')
sys.path.append('../../PointPillars.ops')
from MLP_model import MLPModel
from Dataloader import Dataset
from ultralytics import YOLO
from PointPillars.test import *
from MLP_utils import *


KITTI_TRAIN_PATH = '/media/server1/5150/Wu/KITTI/training'
CKP_PATH = '../../99.pth'
VAL_PATH = '../../ImageSets/val.txt'
RESULT_PATH = '../results/MLP_results/'

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,),(0.5,))])

data_loader_train = Dataset(KITTI_TRAIN_PATH + '/velodyne_reduced', KITTI_TRAIN_PATH + '/calib',
                            KITTI_TRAIN_PATH + '/image_2', KITTI_TRAIN_PATH + '/label_2', val_image_ids=VAL_PATH)

model=MLPModel()
checkpoint = torch.load(CKP_PATH)
model.load_state_dict(checkpoint['model_state_dict'])

optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
optimizer = torch.optim.Adam(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

YOLO_model = YOLO("../../yolov8n.pt")
print('Training epoch {}, training loss'.format(checkpoint['epoch'], checkpoint['loss']))

model.eval()
for idx,((point_file, calib_file), img_file, gt_bbox) in enumerate(tqdm(data_loader_train, desc='Data no.')):
    if VAL_PATH:
        idx = data_loader_train.val_image_ids[idx]
    '''
    YOLO predict
    '''
    YOLO_boxes = YOLO_model.predict(img_file,classes=0)[0].boxes
    YOLO_results = []
    for box in YOLO_boxes:
        if box.cls == 0:
            YOLO_results.append(torch.concat([box.xyxy.squeeze(), box.conf], dim=0).tolist())
    YOLO_results = torch.tensor(YOLO_results)
    # print('YOLO result:',YOLO_results)
    if len(YOLO_results) == 0:
        YOLO_results = torch.tensor([[-1,-1,-1,-1,-1]])
    '''
    PointPillars predict
    '''
    PointPillars_results = PointPillars_test(point_file, calib_file, img_file,
                                             ckpt='/home/server1/Wu/Multi-sensory-fusion/PointPillars/pretrained/epoch_160.pth')
    if len(PointPillars_results) == 0:
        MLP_results = []
    else:
        '''
        Bounding Box Pair
        '''
        input, enclosing_box = align_boxes(YOLO_results, PointPillars_results)
        '''
        Predict
        '''
        predicts = []
        scores = torch.tensor([])
        with torch.no_grad():
            for box_idx, single_input in enumerate(input):
                predict = model(single_input)
                predict, score = torch.split(predict, [4, 1])
                scores = torch.cat([scores, score], dim=0)
                predicts.append(predict)
        scores = torch.sigmoid(scores).numpy()
        predicts = torch.stack(predicts).numpy()

        enclosing_box = enclosing_box[:,:-1].numpy()
        enclosing_box_width = enclosing_box[:,2] - enclosing_box[:,0]
        enclosing_box_height = enclosing_box[:,3] - enclosing_box[:,1]
        enclosing_box_w_h = np.concatenate((enclosing_box_width[np.newaxis,:].T, enclosing_box_height[np.newaxis,:].T),axis=1)

        predict_w_h = enclosing_box_w_h * np.exp(predicts[:,2:])
        predict_center = enclosing_box[:,:2] + enclosing_box_w_h * torch.tensor(predicts[:,0:2]).softmax(dim=1).numpy()
        predict_box = np.concatenate([predict_center - predict_w_h / 2, predict_center + predict_w_h / 2],axis=1)

        MLP_results = np.concatenate([predict_box,scores[np.newaxis,:].T],axis=1)


    with open(RESULT_PATH + f'{str(idx).zfill(6)}.txt', 'w') as f:
        # if len(YOLO_results) == 0:
        #     f.write(' '.join(['DontCare',' '.join(['0' for i in range(15)])]))
        for i in range(len(MLP_results)):
            useless1 = '0.00 0 0.00'
            useless2 = ' '.join(['0.00' for i in range(7)])
            content = ' '.join(['Pedestrian', useless1, ' '.join(['{:.2f}'.format(i) for i in MLP_results[i][:-1]]),
                                useless2, '{:.2f}'.format(MLP_results[i][-1])])
            print(content)
            f.write(content + '\n')
