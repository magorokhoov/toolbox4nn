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
from zmq import device

import data
import modules.networks
import modules.optimizers

from utils import utils


def get_criterion(option_criterion: dict, device=device):
    criterion_type = option_criterion.get('criterion_type')
    reduction = option_criterion.get('reduction', 'mean')

    # Regression #
    if criterion_type in ('MAE', 'l1'):
        criterion_func = nn.L1Loss(reduction=reduction).to(device)
    elif criterion_type in ('MSE', 'l2'):
        criterion_func = nn.MSELoss(reduction=reduction).to(device)
    elif criterion_type in ('elastic'):
        alpha = option_criterion['alpha']
        criterion_func = ElasticLoss(
            alpha=alpha, reduction=reduction, device=device)

    # Classfication #
    elif criterion_type in ('CrossEntropyLoss'):
        criterion_func = nn.CrossEntropyLoss(reduction=reduction).to(device)
    elif criterion_type in ('BCELoss'):
        criterion_func = nn.CrossEntropyLoss(reduction=reduction).to(device)
    elif criterion_type in ('BCEWithLogitsLoss'):
        criterion_func = nn.BCEWithLogitsLoss(reduction=reduction).to(device)
    else:
        raise NotImplementedError(
            f'Loss type [{criterion_type}] is not recognized. losses.py doesn\'t know {[criterion_type]}')
    
    return criterion_func

class ElasticLoss(nn.Module):
    def __init__(self, alpha=0.8, reduction='mean', device='cpu'):
        super(ElasticLoss, self).__init__()

        if alpha < 0.0 or 1.0 < alpha:
            raise Exception("alpha must be from 0.0 to 1.0")

        self.l1_func = nn.L1Loss(reduction=reduction).to(device)
        self.l2_func = nn.MSELoss(reduction=reduction).to(device)

        self.alpha1 = alpha
        self.alpha2 = 1 - alpha
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        l2 = self.l2_func(x, y)
        l1 = self.l1_func(x, y)

        loss = self.alpha1*l1 + self.alpha2*l2
        return loss

class TVLoss(nn.Module):
    def __init__(self, option_criterion: dict) -> None:
        super().__init__()

        self.gamma = option_criterion.get('gamma', 2)

    def forward(self, x):
        b, c, h, w = x.size()
        tv_h = torch.pow(x[:,:,1:,:]-x[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(x[:,:,:,1:]-x[:,:,:,:-1], 2).sum()

        if self.gamma != 2:
            tv = torch.pow(tv_h+tv_w, self.gamma/2.0)
        else:
            tv = tv_h+tv_w

        return tv/(b*c*h*w)
