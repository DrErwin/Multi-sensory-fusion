'''
:Author: Yuhong Wu
:Date: 2023-12-03 15:58:46
:LastEditors: Yuhong Wu
:LastEditTime: 2023-12-09 01:05:38
:Description: 
'''
import math
import os
from pytorch_msssim import ssim, ms_ssim
import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score
import numpy as np
from PIL import Image
from dark_image import *

def rmse(predictions, targets):
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)
    return rmse_val

def psnr(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# psnr_sum = 0
# rmse_sum = 0
# count = 0
# path_1 = r'D:\Multi-sensor_Fusion\Dataset\nirscene1\results\real_B'
# # 大目录，需要从里面挑出来同名的文件
# path_2 = r'D:\Multi-sensor_Fusion\Dataset\nirscene1\results'
# # 备份到的输出路径
# for root, _, files in os.walk(path_1):
#     for file in files:
#         im1 = os.path.join(path_1, file)
#         im2 = os.path.join(path_2+'/pix2pix_L1/test_latest/images', file)
#         im2 = im2.replace('real','fake')
#         im1 = np.array(Image.open(im1))
#         im2 = np.array(Image.open(im2))
#         cur_psnr = psnr(im1, im2)
#         cur_rmse = rmse(im1, im2)
#         psnr_sum += cur_psnr
#         rmse_sum += cur_rmse
#         count += 1
#         # print('psnr:',cur_psnr)
#         # print('rmse:',cur_rmse)
# print(psnr_sum/count)
# print(rmse_sum/count)
trans = transforms.Compose([
            transforms.ToTensor()      # 这里仅以最基本的为例
        ])
degration_cfg=dict(darkness_range=(0.01, 0.4),
                       gamma_range=(2.0, 3.5),
                       rgb_range=(0.8, 0.1),
                       red_range=(1.9, 2.4),
                       blue_range=(1.5, 1.9),
                       quantisation=[4, 6, 8])
trans_to_pil = transforms.ToPILImage(mode="RGB")

im = trans(Image.open(r'C:\Users\Dr.Wu\Desktop\000009.png'))
im2,_ = Low_Illumination_Degrading(im, degration_cfg)
img_pil = trans_to_pil(im2)
img_pil.show()

