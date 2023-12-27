'''
:Author: Yuhong Wu
:Date: 2023-12-03 01:35:59
:LastEditors: Yuhong Wu
:LastEditTime: 2023-12-15 20:15:23
:Description: 
'''
from torchvision.ops.ciou_loss import complete_box_iou_loss
from torchvision.ops.boxes import box_iou
from torch.autograd import Variable
import torch

from torchvision.ops.giou_loss import generalized_box_iou_loss
from torchvision.utils import _log_api_usage_once
from torchvision.ops._utils import _loss_inter_union, _upcast_non_float


THRESHOLD_BBP = 0.7
THRESHOLD_GT = 0.5

'''
:description: 每张图片里的两模态BB坐标及置信度，获得模型输入的14维向量
:param {torch} RGB_BB: (P,5)=(:, [BB,S_c])
:param {torch} LIDAR_BB: (Q,5)=(:, [BB,S_l])
:return {*} 
MLP的输入(Q,14)=(:, [S_c,S_l,BB_c, BB_l, BB_M])
生成的最小外接矩形及其置信度(Q,5)
'''
def align_boxes(RGB_BB:torch.Tensor, LIDAR_BB:torch.Tensor):
    smallest_enclosing_box_list = []
    model_input_list = []
    # 计算每个BB的IOU，匹配阈值THRESHOLD_BBP=0.7
    IOU_metrix = box_iou(RGB_BB[:,:4], LIDAR_BB[:,:4])
    IOU_metrix[IOU_metrix < THRESHOLD_BBP] = 0
    # 对每个LIDAR框找到最佳匹配的RGB框下标
    RGB_index = torch.argmax(IOU_metrix, axis=0) # (Q, )
    for i in range(len(RGB_index)):
        # 有重叠则重叠最大的RGB框和LIDAR的框一起存储，否则只存储LIDAR的框
        if IOU_metrix[RGB_index[i],i] != 0:
            output = torch.cat([RGB_BB[RGB_index[i],-1].unsqueeze(0), LIDAR_BB[i,-1].unsqueeze(0),
                                RGB_BB[RGB_index[i],:4], LIDAR_BB[i,:4]],dim=0)
            enclosing_box = smallest_enclosing_box(RGB_BB[RGB_index[i],:4], LIDAR_BB[i,:4])
            model_input_list.append(torch.cat([output,enclosing_box],dim=0))
            conf = max([RGB_BB[RGB_index[i],-1].unsqueeze(0), LIDAR_BB[i,-1].unsqueeze(0)])
            smallest_enclosing_box_list.append(torch.cat([enclosing_box,conf],dim=0))
            # smallest_enclosing_box_list.append(torch.cat([enclosing_box, torch.tensor(torch.max())],dim=0))
        else:
            output = torch.cat([torch.tensor([-1, LIDAR_BB[i,-1].tolist(), -1, -1, -1, -1]),LIDAR_BB[i,:4]])
            enclosing_box = LIDAR_BB[i,:4]
            model_input_list.append(torch.cat([output,enclosing_box],dim=0))
            conf = LIDAR_BB[i,-1].unsqueeze(0)
            smallest_enclosing_box_list.append(torch.cat([enclosing_box,conf],dim=0))
    
    return torch.stack(model_input_list), torch.stack(smallest_enclosing_box_list)

'''
:description: 
:param {torch} boxes1: (x1, y1, x2, y2)
:param {torch} boxes2: (x1, y1, x2, y2)
:return {*} smallest enclosing box(xc1, yc1, xc2, yc2)
'''
def smallest_enclosing_box(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor):
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(generalized_box_iou_loss)

    boxes1 = _upcast_non_float(boxes1)
    boxes2 = _upcast_non_float(boxes2)

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    # smallest enclosing box
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)
    
    return torch.stack([xc1, yc1, xc2, yc2],dim=0)

'''
:description: 计算模型输出和GTBB的CIOUloss
:param {torch} enclosing_boxes: 输入里匹配到的最小外接矩形坐标及其置信度(Q,5)
:param {torch} predict: 模型的输出(Q,4)=(:, [x_c,y_c,p_w,p_h])
:param {torch} gt_boxes: GT_BB坐标(L,4)
:return {*}
'''
class CIOULoss(torch.nn.Module):
    def __init__(self, threshold_gt=THRESHOLD_GT):
        super(CIOULoss, self).__init__()
        self.threshold_gt = THRESHOLD_GT

    # def forward(self, enclosing_boxes:torch.Tensor, predict:torch.Tensor, gt_boxes:torch.Tensor):
    #     '''
    #     预测BB中心点为(enclosing_box[0] + enclosing_box_width*x_c, enclosing_box[1] + enclosing_box_height*y_c)，
    #     宽度为enclosing_box_width*exp(p_w)，高度为enclosing_box_height*exp(p_h)
    #     '''
    #     IOU_metrix = box_iou(enclosing_boxes[:,:-1], gt_boxes[:,:])
    #     # 计算每个BB的IOU
    #     IOU_metrix[IOU_metrix < self.threshold_gt] = 0
    #     # 对每个最小外接框找到最佳匹配的GT
    #     gt_box_index = torch.argmax(IOU_metrix, axis=1) # (Q, )
    #     loss = Variable(torch.Tensor([0]),requires_grad=True)
    #     for i in range(len(gt_box_index)):
    #         # if IOU_metrix[i,gt_box_index[i]] == 0:
    #         #     continue
    #         gt_box = gt_boxes[gt_box_index[i]]
    #         enclosing_box = enclosing_boxes[i,:-1]
    #         # print(enclosing_box, gt_box)
    #
    #         enclosing_box_width = enclosing_box[2] - enclosing_box[0]
    #         enclosing_box_height = enclosing_box[3] - enclosing_box[1]
    #
    #         enclosing_box_w_h = torch.stack([enclosing_box_width, enclosing_box_height])
    #         predict_w_h = enclosing_box_w_h*torch.exp(predict[i,2:])
    #         predict_center = enclosing_box[:2] + enclosing_box_w_h*predict[i,0:2].softmax(dim=0)
    #         predict_box = torch.cat([predict_center-predict_w_h/2, predict_center+predict_w_h/2])
    #
    #         loss = loss + complete_box_iou_loss(gt_box, predict_box) * ALPHA_CIOU
    #     return loss
    def forward(self, enclosing_box:torch.Tensor, predict:torch.Tensor, gt_boxes:torch.Tensor):
        '''
        预测BB中心点为(enclosing_box[0] + enclosing_box_width*x_c, enclosing_box[1] + enclosing_box_height*y_c)，
        宽度为enclosing_box_width*exp(p_w)，高度为enclosing_box_height*exp(p_h)
        '''
        loss = 0
        IOU_metrix = box_iou(torch.unsqueeze(enclosing_box[:-1],dim=0), gt_boxes[:,:])
        # 计算每个BB的IOU
        IOU_metrix[IOU_metrix < self.threshold_gt] = 0
        # 对每个最小外接框找到最佳匹配的GT
        gt_box_index = torch.argmax(IOU_metrix, axis=1) # (Q, )
        if IOU_metrix[0][gt_box_index] > 0:
            pass
        else:
            return loss, torch.tensor([1])

        gt_box = gt_boxes[gt_box_index]
        enclosing_box = enclosing_box[:-1]
        # print(enclosing_box, gt_box)

        enclosing_box_width = enclosing_box[2] - enclosing_box[0]
        enclosing_box_height = enclosing_box[3] - enclosing_box[1]

        enclosing_box_w_h = torch.stack([enclosing_box_width, enclosing_box_height])
        predict_w_h = enclosing_box_w_h*torch.exp(predict[2:])
        predict_center = enclosing_box[:2] + 2*enclosing_box_w_h*predict[0:2].softmax(dim=0)
        predict_box = torch.cat([predict_center-predict_w_h/2, predict_center+predict_w_h/2])

        loss = loss + complete_box_iou_loss(gt_box, predict_box)
        return loss, torch.tensor([0])


if __name__ == '__main__':
    RGB_BB = torch.tensor([[100,100,200,200,0.7]])
    LIDAR_BB = torch.tensor([[100,110,200,210,0.5],[400,400,500,500,0.3],[90,100,200,200,0.9]])
    a, b = align_boxes(RGB_BB,LIDAR_BB)
    print(a)
    print(b)
    
    # print(calculate_loss(torch.tensor([[100,110,200,210],[400,400,500,500],[90,100,200,200]]),
    #                      torch.tensor([[0.5,0.5,0,0],[0.5,0.5,0,0],[0.5,0.5,0,0]]),
    #                      torch.tensor([100,100,200,200]).unsqueeze(0)))