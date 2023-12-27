import time

from Dataloader import Dataset
from tqdm import tqdm
import sys
sys.path.append('../../PointPillars')
sys.path.append('../../PointPillars.ops')
from PointPillars.test import *


RESULT_PATH = '../results/PointPillars_results/'
KITTI_TRAIN_PATH = '/media/server1/5150/Wu/KITTI/training'
VAL_PATH = '../../ImageSets/val.txt'
data_loader_train = Dataset(KITTI_TRAIN_PATH + '/velodyne_reduced', KITTI_TRAIN_PATH + '/calib',
                            KITTI_TRAIN_PATH + '/image_2', KITTI_TRAIN_PATH + '/label_2', val_image_ids=VAL_PATH)

for idx,((point_file, calib_file), img_file, gt_bbox) in enumerate(tqdm(data_loader_train, desc='Data no.')):
    if VAL_PATH:
        idx = data_loader_train.val_image_ids[idx]
    '''
    PointPillars predict
    '''
    PointPillars_results = PointPillars_test(point_file, calib_file, img_file,
                                             ckpt='/home/server1/Wu/Multi-sensory-fusion/PointPillars/pretrained/epoch_160.pth')

    with open(RESULT_PATH+f'{str(idx).zfill(6)}.txt','w') as f:
        # if len(YOLO_results) == 0:
        #     f.write(' '.join(['DontCare',' '.join(['0' for i in range(15)])]))
        for i in range(len(PointPillars_results)):
            useless1 = '0.00 0 0.00'
            useless2 = ' '.join(['0.00' for i in range(7)])
            content = ' '.join(['Pedestrian',useless1,' '.join(['{:.2f}'.format(i) for i in PointPillars_results[i][:-1]]),
                                useless2,'{:.2f}'.format(PointPillars_results[i][-1])])
            print(content)
            f.write(content+'\n')