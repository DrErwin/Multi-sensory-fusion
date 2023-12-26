'''
:Author: Yuhong Wu
:Date: 2023-12-02 21:42:51
:LastEditors: Yuhong Wu
:LastEditTime: 2023-12-03 03:48:01
:Description:
'''
import numpy as np
import torch


# (S_c, S_l, BB_c, BB_l, BB_M)
# BB = (x1, y1, x2, y2)
N_INPUT = 14
N_HIDDEN_1 = 28
N_HIDDEN_2 = 56
N_HIDDEN_3 = 56
# (t_x, t_y, t_w, t_h, S)
N_OUTPUT = 5


class MLPModel(torch.nn.Module):
    
    def __init__(self):
        super(MLPModel,self).__init__()
        
        self.linear1=torch.nn.Linear(N_INPUT,N_HIDDEN_1, dtype=torch.float64)
        self.linear2=torch.nn.Linear(N_HIDDEN_1,N_HIDDEN_2,dtype=torch.float64)
        self.linear3=torch.nn.Linear(N_HIDDEN_2,N_HIDDEN_3,dtype=torch.float64)
        self.linear4=torch.nn.Linear(N_HIDDEN_3,N_OUTPUT,dtype=torch.float64)
        self.linear4 = torch.nn.Linear(N_HIDDEN_2, N_OUTPUT, dtype=torch.float64)
        self.relu=torch.nn.LeakyReLU()
        self.sigmoid=torch.nn.Sigmoid()
  
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x
 
    