'''
:Author: Yuhong Wu
:Date: 2023-12-09 02:48:17
:LastEditors: Yuhong Wu
:LastEditTime: 2023-12-12 01:02:15
:Description: 
'''
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def save_img(img, path):
    Image.fromarray(img).save(path)
    
def split_and_save(img_file, path=None):
    im = np.array(Image.open(img_file))
    h,w,c = im.shape
    im1 = im[:, :int(w/2), :]
    im1 = np.concatenate((im1,im1),axis=1)
    im2 = im[:, int(w/2):, :]
    im2 = np.concatenate((im2,im2),axis=1)

    save_path, save_name = os.path.split(img_file)
    save_name, save_ext = os.path.splitext(save_name)
    
    save_path = path if path else save_path
    
    save_img(im1, save_path+save_name+'_01'+save_ext)
    save_img(im2, save_path+save_name+'_02'+save_ext)

def resize(width,height,img):
    img = Image.open(img)
    return img.resize((width, height),Image.Resampling.LANCZOS)

def concat_and_save(im1_file, im2_file):
    # h,w,c = np.array(Image.open(im1_file)).shape
    # im1 = resize(int(w/2), h, im1_file)
    # im2 = resize(int(w/2), h, im2_file)
    im1 = np.array(Image.open(im1_file))
    im2 = np.array(Image.open(im2_file))
    h,w,c = im1.shape
    im1 = im1[:, :int(w/2), :]
    im2 = im2[:, :int(w/2), :]
    new_img = np.concatenate((im1,im2), axis=1)
    
    save_path, save_name = os.path.split(im1_file)
    save_name, save_ext = os.path.splitext(save_name)
    save_name = save_name.split('_')[0] + '_new'
    save_img(new_img, save_path+save_name+save_ext)
    
# split_and_save('000009.png')
concat_and_save('000009_01.png', '000009_02.png')

