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

import data
import modules.networks
import modules.optimizers

from utils import utils


def get_criterion(option_criterion: dict):
    criterion_type = option_criterion.get('criterion_type')
    reduction = option_criterion.get('reduction', 'mean')

    # Regression #
    if criterion_type in ('MAE', 'l1'):
        criterion_func = nn.L1Loss(reduction=reduction)
    elif criterion_type in ('MSE', 'l2'):
        criterion_func = nn.MSELoss(reduction=reduction)
    elif criterion_type in ('elastic'):
        alpha = option_criterion['alpha']
        criterion_func = ElasticLoss(
            alpha=alpha, reduction=reduction)

    # Classfication #
    elif criterion_type in ('CrossEntropyLoss'):
        criterion_func = nn.CrossEntropyLoss(reduction=reduction)
    elif criterion_type in ('BCELoss'):
        criterion_func = nn.CrossEntropyLoss(reduction=reduction)
    elif criterion_type in ('BCEWithLogitsLoss'):
        criterion_func = nn.BCEWithLogitsLoss(reduction=reduction)
    else:
        raise NotImplementedError(
            f'Loss type [{criterion_type}] is not recognized. losses.py doesn\'t know {[criterion_type]}')
    
    return criterion_func

class ElasticLoss(nn.Module):
    def __init__(self, alpha=0.2, reduction='mean'):
        super(ElasticLoss, self).__init__()

        if alpha < 0.0 or 1.0 < alpha:
            raise Exception("alpha must be from 0.0 to 1.0")

        self.alpha1 = torch.FloatTensor(alpha)
        self.alpha2 = torch.FloatTensor(1 - alpha)
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        l2 = F.mse_loss(x.squeeze(), y.squeeze(),
                        reduction=self.reduction).mul(self.alpha[0])
        l1 = F.l1_loss(x.squeeze(), y.squeeze(),
                       reduction=self.reduction).mul(self.alpha[1])

        loss = l1 + l2
        return loss

class TVLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
