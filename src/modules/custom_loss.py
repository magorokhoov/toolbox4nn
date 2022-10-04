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

import data
import modules.networks as networks
import modules.optimizers

from utils import utils


def get_criterion(option_criterion: dict, device):
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
            f'Loss type [{criterion_type}] is not recognized in losser.py')
    
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

class PerceptualLoss(nn.Module):
    loss_models = nn.ModuleDict()

    def __init__(self, loss_params: dict, device) -> None:
        super().__init__()

        self.network_name = loss_params['network']
        self.criterion = get_criterion(loss_params['criterion'], device=device)
        self.layers_dict = loss_params['layers']
        self.layers_list = list(self.layers_dict)

        if self.network_name not in PerceptualLoss.loss_models:
            model = networks.get_network({'arch': self.network_name}, device=device)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            PerceptualLoss.loss_models[self.network_name] = model

        self.mean_val = torch.tensor(
                [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False).to(device=device)
        self.std_val = torch.tensor(
                [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False).to(device=device)

    def forward(self, x:torch.Tensor, target:torch.Tensor):
        #x = x.float()
        #target = target.float()

        #with torch.no_grad():

        
        if self.network_name in ('vgg16', 'vgg19'):
            x = (x - self.mean_val) / self.std_val
            target = (target - self.mean_val) / self.std_val

        feas_x = PerceptualLoss.loss_models[self.network_name](
            x,
            listen_list=self.layers_list
        )
        feas_target = PerceptualLoss.loss_models[self.network_name](
            target.detach(),
            listen_list=self.layers_list
        )

        loss = 0
        for layer_name in self.layers_list:
            loss += self.layers_dict[layer_name] * self.criterion(
                feas_x[layer_name],
                feas_target[layer_name]
            )

        return loss

