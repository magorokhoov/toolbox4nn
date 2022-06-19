import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np

class my_simple_CLNN(nn.Module):
    def __init__(self, option_arch:dict):
        super(my_simple_CLNN, self).__init__()

        in_nc = option_arch.get('in_nc')
        mid_nc = option_arch.get('mid_nc')
        out_nc = option_arch.get('out_nc')
        
        pad3 = nn.ReflectionPad2d(3)
        act = nn.LeakyReLU(0.2, inplace=True)
        conv0 = nn.Conv2d(in_nc, 2*mid_nc, kernel_size=7, stride=2) # 64
        conv1 = nn.Conv2d(2*mid_nc, 2*mid_nc, kernel_size=3, stride=1, padding=1) # 64
        conv2 = nn.Conv2d(2*mid_nc, 4*mid_nc, kernel_size=3, stride=2, padding=1) # 32
        conv3 = nn.Conv2d(4*mid_nc, 8*mid_nc, kernel_size=3, stride=1, padding=1) # 32
        conv4 = nn.Conv2d(8*mid_nc, 2*mid_nc, kernel_size=3, stride=2, padding=1) # 16
        avgpool = nn.AvgPool2d((2,2)) # 8
        
        flatten = nn.Flatten()
        linear0 = nn.Linear(8*8*2*mid_nc, 8*mid_nc)
        linear1 = nn.Linear(8*mid_nc, 2*mid_nc)
        linear2 = nn.Linear(2*mid_nc, out_nc)
        
        features = [pad3, conv0, act, conv1, act, conv2, act, conv3, act, conv4, avgpool]
        classificator = [flatten, linear0, act, linear1, act, linear2]
        
        self.features = nn.Sequential(*features)
        self.classificator = nn.Sequential(*classificator)
    
    def forward(self, x):
        fea = self.features(x)
        #b, c, h, w = fea.shape
        #print(c,h,w)
        out = self.classificator(fea)
        return out
    
    def print_num_parameters(self):
        num_params_features = sum(p.numel() for p in self.features.parameters())
        num_params_classficator = sum(p.numel() for p in self.classificator.parameters())
        
        print(f'Features params: {num_params_features}')
        print(f'Classificator params: {num_params_classficator}')
        
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())