'''
:Author: Yuhong Wu
:Date: 2023-12-03 01:33:23
:LastEditors: Yuhong Wu
:LastEditTime: 2023-12-12 16:05:17
:Description: 
'''
import pathlib
import re
from typing import Iterator
import numpy as np
from torch.utils.data import DataLoader, IterableDataset

def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    annotations['name'] = np.array([x[0] for x in content])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array(
        [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array(
        [[float(info) for info in x[8:11]] for x in content]).reshape(
            -1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array(
        [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array(
        [float(x[14]) for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros([len(annotations['bbox'])])
    return annotations


class Dataset(IterableDataset):
    def __init__(self, point_cloud_path, calib_path, img_path, label_path) -> None:
        super().__init__()
        self.label_path = label_path
        self.img_path = img_path
        self.point_cloud_path = point_cloud_path
        self.calib_path = calib_path
        self.annos = []
    '''
    Return: (path_of_point_file, path_of_calib_file), path_of_img_file, labels_in_an_img
    '''
    def read_data(self):
        filepaths = pathlib.Path(self.label_path).glob('*.txt')
        prog = re.compile(r'^\d{6}.txt$')
        filepaths = filter(lambda f: prog.match(f.name), filepaths)
        image_ids = [int(p.stem) for p in filepaths]
        image_ids = sorted(image_ids)
        
        label_path = pathlib.Path(self.label_path)
        img_path = pathlib.Path(self.img_path)
        point_cloud_path = pathlib.Path(self.point_cloud_path)
        calib_path = pathlib.Path(self.calib_path)

        for idx in image_ids:
            image_idx = "{:06d}".format(idx)
            label_filename = label_path / (image_idx + '.txt')
            cur_anno = get_label_anno(label_filename)
            self.annos.append(cur_anno)
            
            label = cur_anno['bbox'][cur_anno['name'] == 'Pedestrian']
            img = img_path / (image_idx + '.png')
            point_cloud = point_cloud_path / (image_idx + '.bin')
            calib = calib_path / (image_idx + '.txt')
            
            yield (point_cloud, calib), img, label

    def __iter__(self) -> Iterator:
        return self.read_data()
    
