import argparse
import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO

from utils import setup_seed, read_points, read_calib, read_label, \
    keep_bbox_from_image_range, keep_bbox_from_lidar_range, vis_pc, \
    vis_img_3d, vis_img_2d, bbox3d2corners_camera, points_camera2image, \
    bbox_camera2lidar
from model import PointPillars


def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts 


def main(args):
    CLASSES = {
        'Pedestrian': 0,
        'Cyclist': 1,
        'Car': 2
    }
    LABEL2CLASSES = {v: k for k, v in CLASSES.items()}
    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

    if not args.no_cuda:
        model = PointPillars(nclasses=len(CLASSES)).cuda()
        model.load_state_dict(torch.load(args.ckpt))
    else:
        model = PointPillars(nclasses=len(CLASSES))
        model.load_state_dict(
            torch.load(args.ckpt, map_location=torch.device('cpu')))

    if not os.path.exists(args.pc_path):
        raise FileNotFoundError
    pc = read_points(args.pc_path)
    pc = point_range_filter(pc)
    pc_torch = torch.from_numpy(pc)
    if os.path.exists(args.calib_path):
        calib_info = read_calib(args.calib_path)
    else:
        calib_info = None

    if os.path.exists(args.gt_path):
        gt_label = read_label(args.gt_path)
    else:
        gt_label = None

    if os.path.exists(args.img_path):
        img = cv2.imread(args.img_path, 1)
    else:
        img = None

    model.eval()
    with torch.no_grad():
        if not args.no_cuda:
            pc_torch = pc_torch.cuda()

        result_filter = model(batched_pts=[pc_torch],
                              mode='test')[0]
    if calib_info is not None and img is not None:
        tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)
        P2 = calib_info['P2'].astype(np.float32)

        image_shape = img.shape[:2]
        result_filter = keep_bbox_from_image_range(result_filter, tr_velo_to_cam, r0_rect, P2, image_shape)

    result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
    lidar_bboxes = result_filter['lidar_bboxes']
    labels, scores = result_filter['labels'], result_filter['scores']
    bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes']

    pedestrain_bboxes_2d = bboxes2d[labels == 0]
    print(pedestrain_bboxes_2d)

    yolo_model = YOLO('yolov8s.pt')
    yolo_results = yolo_model('../KITTI_img/')
    boxes = yolo_results[0].boxes
    for i in boxes:
        if i.cls == 0:
            print(i.xyxy.tolist())
            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--ckpt', default='pretrained/epoch_160.pth', help='your checkpoint for kitti')
    parser.add_argument('--pc_path', help='your point cloud path')
    parser.add_argument('--calib_path', default='', help='your calib file path')
    parser.add_argument('--gt_path', default='', help='your ground truth path')
    parser.add_argument('--img_path', default='', help='your image path')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    parser.add_argument('--vis_dimen', default='2d', help='Bounding box in image: [2d,3d]')
    parser.add_argument('--vis_pc', default=False, help='Visualize point cloud')
    args = parser.parse_args()

    main(args)
