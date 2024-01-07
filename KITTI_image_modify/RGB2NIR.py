import glob
import os

from tqdm import tqdm

import KITTI_img_split

RGB_PATH = '/media/server1/5150/Wu/KITTI/training/image_2'
SPLIT_PATH = '/media/server1/5150/Wu/KITTI/training/image_2_split/test'
RGB_list = glob.glob(os.path.join(RGB_PATH,'*.png'))

for i in tqdm(RGB_list):
    KITTI_img_split.split_and_save(i,SPLIT_PATH)