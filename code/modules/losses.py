# https://github.com/magorokhoov
# 11jun2022sat
# Mikhail Gorokhov
# Coding custom NN toolbox

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import argparse
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt


from tqdm import tqdm
from tqdm import tqdm_notebook

from utilsss.utilsss import *
import modules.custom_loss as custom_loss


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

        self.n_accumulation = 0.000001 # to prevent division by zero
        self.losses_accumulation = {}

    def reset_accumulation(self) -> None:
        for loss_name in self.losses_accumulation:
            self.losses_accumulation[loss_name] = 0.0

        self.n_accumulation = 0.000001 # to prevent division by zero

    def add_accumulation(self, add_losses:dict) -> None:
        self.n_accumulation += 1
        for loss_name in add_losses:
            #print(add_losses)
            self.losses_accumulation[loss_name] += add_losses[loss_name]

    def get_accumulation(self) -> dict:
        return self.losses_accumulation.copy()
    
    def get_losses_str(self, reset=True) -> str:
        losses_str = ''
        for i, loss_name in enumerate(self.losses_accumulation):
            mean_loss = self.losses_accumulation[loss_name]/self.n_accumulation
            losses_str += f'{loss_name:s}: {mean_loss:.4e}'
            if i < len(self.losses_accumulation) - 1:
                losses_str += ', '

        if reset:
            self.reset_accumulation()

        return losses_str

    def forward(self):
        pass
    


class ClassificatorLoss(BaseLoss):
    def __init__(self, option_loss: dict):
        super(ClassificatorLoss, self).__init__()

        func_type = option_loss.get('func_type', None)
        weight = option_loss.get('weight', 1.0)
        reduction = option_loss.get('reduction', 'mean')

        if func_type is None:
            raise NameError('func_type cannot be None. Set CrossEntropyLoss or something else')

        self.func = get_loss_func(func_type, weight, reduction=reduction)

        self.losses_accumulation[self.func['func_type']] = 0.0

    def forward(self, pred:torch.Tensor, target:torch.Tensor):
        loss = self.func['func'](pred, target) * self.func['weight']

        self.add_accumulation({self.func['func_type']: loss.item()})

        return loss
        

'''
class LossStats:
    def __init__(self):
        pass
    def 

class ClassificatorLoss:
    def __init__(self, option_loss: dict):
        #self.option_loss = option_loss

        func_type = option_loss.get('func_type', None)
        weight = option_loss.get('weight', 1.0)
        reduction = option_loss.get('reduction', 'mean')

        if func_type is None:
            raise NameError('func_type cannot be None. Set CrossEntropyLoss or something else')
        
        self.last_stats = {}
        self.funcs = []

        func = get_loss_func(func_type, weight, reduction=reduction)
        self.funcs += [func] 

    def calc_total_loss(self, pred, target):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f'pred not right type. Type of target is {type(target)}')
        if not isinstance(target, torch.Tensor):
            raise TypeError(f'target not right type. Type of target is {type(target)}')

        total_loss = 0.0
        for func in self.funcs:
            loss = func['func'](pred, target) * func['weight']
            total_loss += loss
            self.last_stats[func['func_type']] = loss.item()

        self.last_stats['total_loss'] = total_loss.item()

        return total_loss

    def get_last_stats(self) -> dict:
        return self.last_stats.copy()
'''
def get_loss_func(func_type, weight, reduction='mean', neural_network=None, alpha=0.2, option_loss=None):
    
    ### Regression ###
    if func_type in ('MAE', 'l1'):
        loss_function = nn.L1Loss(reduction=reduction)
    elif func_type in ('MSE', 'l2'):
        loss_function = nn.MSELoss(reduction=reduction)
    elif func_type in ('elastic'):
        loss_function = custom_loss.ElasticLoss(alpha=alpha, reduction=reduction)

    ### Classfication ###
    elif func_type in ('CrossEntropyLoss'):
        loss_function = nn.CrossEntropyLoss(reduction=reduction)
    elif func_type in ('BCELoss'):
        loss_function = nn.CrossEntropyLoss(reduction=reduction)
    elif func_type in ('BCEWithLogitsLoss'):
        loss_function = nn.BCEWithLogitsLoss(reduction=reduction)
    else:
        raise NotImplementedError(f'Loss type [{name}] is not recognized. losses.py doesn\'t know {[name]}')

    return {'func_type': func_type,
            'weight': weight,
            'func': loss_function
    }

    

'''
class Losser: # but not loser...
    def __init__(self, option_loss: dict):
        self.loss_list = self.get_loss_list(option_loss)
        self.loss_log_dict = {}

    def get_loss_results(self, x:torch.Tensor, y:torch.Tensor):

        loss_result = 0.0
        for loss in self.loss_list:
            #print(loss)
            #print(loss['weight'], type(loss['weight']))
            current_loss = loss['weight']*loss['function'](x, y)
            loss_result += current_loss
            self.loss_log_dict[loss['loss_type']] = current_loss.item()

        return loss_result

    def get_loss_list(self, option_loss: dict):
        pixel_criterion = option_loss.get('pixel_criterion', None)
        pixel_weight = option_loss.get('pixel_weight', 0.0)
        pixel_alpha = option_loss.get('pixel_alpha', 0.0)


        loss_list = []

        if pixel_criterion is not None and pixel_weight > 0.0:
            pixel_loss = self.get_loss_func(pixel_criterion, pixel_weight, alpha=pixel_alpha)
            loss_list += [pixel_loss]

        return loss_list
'''