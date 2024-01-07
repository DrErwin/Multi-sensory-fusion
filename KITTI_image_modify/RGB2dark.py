from tqdm import tqdm

import dark_image
from PIL import Image
import glob
import os
from torchvision.transforms import transforms

RGB_PATH = '/media/server1/5150/Wu/KITTI/training/image_2'
DARK_PATH = '/media/server1/5150/Wu/KITTI/training/image_2_dark'
transform = transforms.Compose([transforms.ToTensor()])

RGB_list = glob.glob(os.path.join(RGB_PATH,'*.png'))
for i in tqdm(RGB_list):
    img = Image.open(i)
    img = transform(img)
    img_dark,_ = dark_image.Low_Illumination_Degrading(img,dark_image.COCO_CFG)
    img_dark = transforms.ToPILImage()(img_dark)
    img_dark.save(os.path.join(DARK_PATH, os.path.split(i)[1]))

