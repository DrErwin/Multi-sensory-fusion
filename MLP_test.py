import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
from model import MLPModel
from Dataloader import Dataset
from ultralytics import YOLO
from .PointPillars.test import *
from utils import *

BATCH_SIZE = 64
EPOCH = 100
CKP_PATH = './PointPillars/pretrained/epoch_160.pth'

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,),(0.5,))])

data_loader_test = Dataset('','','','')
checkpoint = torch.load(CKP_PATH)
model=MLPModel().load_state_dict(checkpoint['model_state_dict'])
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
YOLO_model = YOLO("yolov8n.pt")
print('Training epoch {}, training loss'.format(checkpoint['epoch'], checkpoint['loss']))

model.eval()

for (point_file, calib_file), img_file, gt_boxes in data_loader_test:
    labels = torch.tensor(labels)
    '''
    YOLO predict
    '''
    YOLO_boxes = YOLO_model([img_file]).boxes
    YOLO_results = []
    for box in YOLO_boxes:
        if box.cls == 0:
            YOLO_results.append(torch.concat([box.xyxy.squeeze(), box.conf], dim=0).tolist())
    YOLO_results = torch.tensor(YOLO_results)
    
    '''
    PointPillars predict
    '''
    PointPillars_results = PointPillars_test(point_file, calib_file, img_file)

    '''
    Bounding Box Pair
    '''
    input, enclosing_box = align_boxes(YOLO_results, PointPillars_results)
    enclosing_box, conf = torch.split(enclosing_box, (4,1), dim=1)
    
    '''
    Predict
    '''
    predict = model(input)
    enclosing_box_width = enclosing_box[:,2] - enclosing_box[:,0]
    enclosing_box_height = enclosing_box[:,3] - enclosing_box[:,1]
    enclosing_box_w_h = torch.concat([enclosing_box_width, enclosing_box_height],dim=0)
    predict_w_h = enclosing_box_w_h*torch.exp(predict[:,2:])
    predict_center = enclosing_box[:,:2] + enclosing_box_w_h*predict[:,:2].softmax(dim=0)
    predict_box = torch.cat([predict_center-predict_w_h/2, predict_center+predict_w_h/2])
