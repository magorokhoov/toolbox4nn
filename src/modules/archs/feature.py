import torch
import torch.nn as nn
import torch.nn.functional as F

from . import blocks

class VGG16_fea(nn.Module):
    def __init__(self):
        super().__init__()
        
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        act = nn.ReLU(inplace=True)
        
        conv1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        conv1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        
        conv2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        conv2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        
        conv3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        conv3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        conv3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        
        conv4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        conv4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        conv4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        
        conv5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        conv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
    
    
        self.features = nn.Sequential(
            conv1_1, act, conv1_2, act, maxpool,
            conv2_1, act, conv2_2, act, maxpool,
            conv3_1, act, conv3_2, act, conv3_3, act, maxpool,
            conv4_1, act, conv4_2, act, conv4_3, act, maxpool,
            conv5_1, act, conv5_2, act, conv5_3, # act, maxpool, <- not needed for feas
        )
        
        self.dict_layer2index = {
            'conv1_1': 0, 'conv1_2': 2,
            'conv2_1': 5, 'conv2_2': 7,
            'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14, 
            'conv4_1': 17, 'conv4_2': 19, 'conv4_3': 21, 
            'conv5_1': 24, 'conv5_2': 26, 'conv4_3': 28, 
        }
        self.dict_index2layer = \
            dict([(v, k) for k, v in self.dict_layer2index.items()])
        
    def forward(self, x, listen_list:list) -> dict:
        assert len(listen_list) >= 1, 'listen_list must >= 1'

        listen_indexes = []
        for i in range(len(listen_list)):
            listen_indexes += [self.dict_layer2index[listen_list[i]]] 
        max_index = max(listen_indexes)
        
        feas = {}
        fea = x
        for i in range(max_index+1):
            fea = self.features[i](fea)
            if i in listen_indexes:
                layer_name = self.dict_index2layer[i]
                feas[layer_name] = fea
        
        return feas
        
        
class VGG19_fea(nn.Module):
    def __init__(self):
        super().__init__()
        
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        act = nn.ReLU(inplace=True)
        
        conv1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        conv1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        
        conv2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        conv2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        
        conv3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        conv3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        conv3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        conv3_4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        
        conv4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        conv4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        conv4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        conv4_4 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        
        conv5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        conv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        conv5_4 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
    
    
        self.features = nn.Sequential(
            conv1_1, act, conv1_2, act, maxpool,
            conv2_1, act, conv2_2, act, maxpool,
            conv3_1, act, conv3_2, act, conv3_3, act, conv3_4, act, maxpool,
            conv4_1, act, conv4_2, act, conv4_3, act, conv4_4, act, maxpool,
            conv5_1, act, conv5_2, act, conv5_3, act, conv5_4, # act, maxpool, <- not needed for feas
        )
        
        self.dict_layer2index = {
            'conv1_1': 0, 'conv1_2': 2,
            'conv2_1': 5, 'conv2_2': 7,
            'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14, 'conv3_4': 16, 
            'conv4_1': 19, 'conv4_2': 21, 'conv4_3': 23, 'conv4_4': 25, 
            'conv5_1': 28, 'conv5_2': 30, 'conv4_3': 32, 'conv5_4': 34, 
        }
        self.dict_index2layer = \
            dict([(v, k) for k, v in self.dict_layer2index.items()])

    def forward(self, x, listen_list:list) -> dict:
        assert len(listen_list) >= 1, 'listen_list must >= 1'

        listen_indexes = []
        for i in range(len(listen_list)):
            listen_indexes += [self.dict_layer2index[listen_list[i]]] 
        max_index = max(listen_indexes)
        
        feas = {}
        fea = x
        for i in range(max_index+1):
            fea = self.features[i](fea)
            if i in listen_indexes:
                layer_name = self.dict_index2layer[i]
                feas[layer_name] = fea
        
        return feas
        