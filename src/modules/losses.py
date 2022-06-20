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

from utils import utils
import modules.custom_loss as custom_loss


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

        self.n_accumulation = 0.000001  # to prevent division by zero
        self.losses_accumulation = {}

    def reset_accumulation(self) -> None:
        for loss_name in self.losses_accumulation:
            self.losses_accumulation[loss_name] = 0.0

        self.n_accumulation = 0.000001  # to prevent division by zero

    def add_accumulation(self, add_losses: dict) -> None:
        self.n_accumulation += 1
        for loss_name in add_losses:
            # print(add_losses)
            self.losses_accumulation[loss_name] += add_losses[loss_name]

    def get_accumulation(self) -> dict:
        return self.losses_accumulation.copy()

    def get_losses_str(self, reset=True) -> str:
        losses_str = ''
        for i, loss_name in enumerate(self.losses_accumulation):
            mean_loss = self.losses_accumulation[loss_name] / \
                self.n_accumulation
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
            raise NameError(
                'func_type cannot be None. Set CrossEntropyLoss or something else')

        self.func = get_loss_func(func_type, weight, reduction=reduction)

        self.losses_accumulation[self.func['func_type']] = 0.0

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        loss = self.func['func'](pred, target) * self.func['weight']

        self.add_accumulation({self.func['func_type']: loss.item()})

        return loss


def get_loss_func(
        func_type,
        weight,
        reduction='mean',
        neural_network=None,
        alpha=0.2,
        option_loss=None):

    ### Regression ###
    if func_type in ('MAE', 'l1'):
        loss_function = nn.L1Loss(reduction=reduction)
    elif func_type in ('MSE', 'l2'):
        loss_function = nn.MSELoss(reduction=reduction)
    elif func_type in ('elastic'):
        loss_function = custom_loss.ElasticLoss(
            alpha=alpha, reduction=reduction)

    ### Classfication ###
    elif func_type in ('CrossEntropyLoss'):
        loss_function = nn.CrossEntropyLoss(reduction=reduction)
    elif func_type in ('BCELoss'):
        loss_function = nn.CrossEntropyLoss(reduction=reduction)
    elif func_type in ('BCEWithLogitsLoss'):
        loss_function = nn.BCEWithLogitsLoss(reduction=reduction)
    else:
        raise NotImplementedError(
            f'Loss type [{func_type}] is not recognized. losses.py doesn\'t know {[func_type]}')

    return {'func_type': func_type,
            'weight': weight,
            'func': loss_function
            }
            