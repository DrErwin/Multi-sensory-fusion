'''
:Author: Yuhong Wu
:Date: 2023-12-02 21:44:31
:LastEditors: Yuhong Wu
:LastEditTime: 2023-12-15 20:14:55
:Description: 
'''
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

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,),(0.5,))])

data_loader_train = Dataset('','','','')
model=MLPModel()
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
YOLO_model = YOLO("yolov8n.pt")

model.train()
for epoch in range(EPOCH) :
    sum_loss=0
    train_correct=0
    idx = 0
    for (point_file, calib_file), img_file, gt_boxes in data_loader_train:
        idx += 1
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
        
        '''
        Predict and compute loss
        '''
        predict = model(input)
        optimizer.zero_grad()
        loss = calculate_loss(enclosing_box, predict, gt_boxes)
        loss.backward()
        optimizer.step()
 
        sum_loss+=loss.data
    print('epoch[%d/%d] loss:%.03f' % (epoch + 1, EPOCH, sum_loss / idx))
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    
    if epoch // 5 == 0:
        torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, f'{epoch}.pth')
    
# model.eval()
# test_correct = 0
# for data in data_loader_test :
#     inputs, lables = data
#     inputs, lables = Variable(inputs).cpu(), Variable(lables).cpu()
#     inputs=torch.flatten(inputs,start_dim=1) #展并数据
#     outputs = model(inputs)
#     _, id = torch.max(outputs.data, 1)
#     test_correct += torch.sum(id == lables.data)
# print("correct:%.3f%%" % (100 * test_correct / len(data_test )))