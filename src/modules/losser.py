# https://github.com/magorokhoov
# 11jun2022sat
# Mikhail Gorokhov
# Coding custom NN toolbox

from matplotlib.style import available
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

from utils import utils
import modules.custom_loss as custom_loss


def get_losser(losser_type, option_loss: dict, item_freq: int=1):
    if losser_type is None:
        raise NotImplementedError(
            'losser_type is None. Please, add losser_type to config file')

    losser_type = losser_type.lower()
    if losser_type in ('class', 'image'):
        losser = Losser(option_loss, item_freq=item_freq)
    else:
        raise NotImplementedError(
            f'losser_type [{losser_type}] is not implemented')

    return losser


class Losser(nn.Module):
    def __init__(self, option_loss: dict, item_freq: int=1):
        super(Losser, self).__init__()
        self.option_loss = option_loss

        self.loss_funcs = []
        for loss_name in self.option_loss:
            loss_params = self.option_loss.get(loss_name)
            loss_type = self.option_loss['type']
            func = get_loss_func(loss_name=loss_name, loss_type=loss_type, loss_params=loss_params)

            self.loss_funcs += [func]

        self.n_accumulation = 0.000001
        self.accumulation = {}
        self.iter = 0
        self.item_freq = item_freq

        for func in self.loss_funcs:
            loss_name = func['loss_name']
            self.accumulation[loss_name] = 0.0

        self.reset_accumulation()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):

        losses_dict = {}
        total_loss = 0.0
        for func in self.loss_funcs:
            loss_type = func.get('loss_type')

            if loss_type in ('pixel', 'class'):
                loss = func['func'](pred, target)

            elif loss_type in ('tv'):
                loss = func['func'](pred)

            else:
                raise NotImplementedError(
                    f'loss_type {loss_type} is not implemented in Losser forward')

            loss *= func['weight']

            if self.iter % self.item_freq == 0:
                losses_dict[func['loss_name']] = loss.item()

            total_loss += loss
            
        if self.iter % self.item_freq == 0:
            self.add_accumulation(add_losses=losses_dict)

        return total_loss

    
    def funcs_to_cuda(self) -> None:
        for i in range(len(self.loss_funcs)):
            self.loss_funcs[i]['func'] = self.loss_funcs[i]['func'].to('cuda')
            # print(self.loss_funcs[i]['func'])

    def reset_accumulation(self) -> None:
        for accumulation_name in self.accumulation:
            self.accumulation[accumulation_name] = 0.0

        self.n_accumulation = 0.000001  # to prevent division by zero

    def add_accumulation(self, add_losses: dict) -> None:
        self.n_accumulation += 1
        for loss_name in add_losses:
            # print(add_losses)
            self.accumulation[loss_name] += add_losses[loss_name]

    def get_accumulation(self) -> dict:
        return self.accumulation.copy()

    def get_losses_str(self, reset=True) -> str:
        losses_str = ''
        for i, loss_name in enumerate(self.accumulation):
            mean_loss = self.accumulation[loss_name] / \
                self.n_accumulation
            losses_str += f'{loss_name:s}: {mean_loss:.4e}'
            if i < len(self.accumulation) - 1:
                losses_str += ', '

        if reset:
            self.reset_accumulation()

        return losses_str


def get_loss_func(loss_name: str, loss_type: str, loss_params: dict) -> dict:
    option_criterion = loss_params.get('criterion')
    
    if loss_type == 'class':
        loss_func = custom_loss.get_criterion(
            option_criterion=option_criterion)
    elif loss_type == 'pixel':
        loss_func = custom_loss.get_criterion(
            option_criterion=option_criterion)
    elif loss_type == 'tv':
        loss_func = custom_loss.TVLoss(option_criterion=option_criterion)
    else:
        raise NotImplementedError(f'loss_type {loss_type} is not implemented')

    weight = loss_params['weight']

    return {'loss_name': loss_name,
            'loss_type': loss_type,
            'weight': weight,
            'func': loss_func
            }


'''
class BaseLosser(nn.Module):
    def __init__(self, option_loss: dict) -> None:
        super(BaseLosser, self).__init__()

        self.option_loss = option_loss
        self.n_accumulation = 0.000001
        self.accumulation = {}

        for func in self.funcs:
            loss_name = self.func['loss_name']
            self.accumulation[loss_name] = 0.0

        self.reset_accumulation()

    def reset_accumulation(self) -> None:
        for accumulation_name in self.accumulation:
            self.accumulation[loss_name] = 0.0

        self.n_accumulation = 0.000001  # to prevent division by zero

    def add_accumulation(self, add_losses: dict) -> None:
        self.n_accumulation += 1
        for loss_name in add_losses:
            # print(add_losses)
            self.accumulation[loss_name] += add_losses[loss_name]

    def get_accumulation(self) -> dict:
        return self.accumulation.copy()

    def get_losses_str(self, reset=True) -> str:
        losses_str = ''
        for i, loss_name in enumerate(self.accumulation):
            mean_loss = self.accumulation[loss_name] / \
                self.n_accumulation
            losses_str += f'{loss_name:s}: {mean_loss:.4e}'
            if i < len(self.accumulation) - 1:
                losses_str += ', '

        if reset:
            self.reset_accumulation()

        return losses_str

    def forward(self) -> None:
        raise NotImplementedError(
            'forward of BaseLosser is not Implemented. Please, use concrete losser instand')

class ClassificatorLosser(BaseLosser):
    def __init__(self, option_loss: dict):
        super(ClassificatorLosser, self).__init__()

        func_type = option_loss['func_type']
        weight = option_loss.get('weight', 1.0)
        reduction = option_loss.get('reduction', 'mean')

        self.func = get_loss_func()

        self.accumulation[self.func['loss_name']] = 0.0

    def forward(self, pred: torch.Tensor, target: torch.Tensor):

        loss = self.func['func'](pred, target) * self.func['weight']

        self.add_accumulation({self.func['loss_name']: loss.item()})

        return loss


class ImageLosser(BaseLosser):
    def __init__(self, option_loss: dict):
        super(ImageLosser, self).__init__()

        self.funcs = {}
        for loss_option_param in option_loss:
            if loss_option_param == 'pixel':
                func_type = option_loss['func_type']
                weight = option_loss.get('weight', 1.0)
                reduction = option_loss.get('reduction', 'mean')

                loss_name = 'pixel_' + func_type
                self.funcs[loss_name] = get_loss_func(
                    loss_name=loss_name, func_type=func_type, weight=weight, reduction=reduction)

            self.accumulation[loss_name] = 0.0

    def forward(self, x, y=None):
        return 
'''
