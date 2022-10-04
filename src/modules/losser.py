# https://github.com/magorokhoov
# 11jun2022sat
# Mikhail Gorokhov
# Coding custom NN toolbox

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import yaml
import numpy as np
import cv2

from utils import utils
import modules.custom_loss as custom_loss


def get_losser(option_losser: dict, device):
    losser_type = option_losser['losser_type']
    option_losses = option_losser['losses']

    if losser_type is None:
        raise NotImplementedError(
            'losser_type is None. Please, add losser_type to config file')

    losser_type = losser_type.lower()
    if losser_type in ('class', 'image'):
        losser = Losser(option_losses=option_losses, device=device)
    else:
        raise NotImplementedError(
            f'losser_type [{losser_type}] is not implemented')

    return losser
    

class Losser(nn.Module):
    def __init__(self, option_losses: dict, device) -> None:
        super().__init__()

        self.option_losses = option_losses
        self.last_losses = {}

        self.loss_funcs = [] # nn.ModuleList()
        for loss_name in self.option_losses:
            loss_params = self.option_losses[loss_name]
            loss_type = loss_params['loss_type']
            func = get_loss_func(loss_name=loss_name, loss_type=loss_type, loss_params=loss_params, device=device)

            self.loss_funcs += [func]

    def forward(self, pred: torch.Tensor, target: torch.Tensor):

        total_loss = 0.0
        for func in self.loss_funcs:
            loss_type = func['loss_type']

            if loss_type in ('pixel', 'class', 'perceptual'):
                loss = func['func'](pred, target)

            elif loss_type in ('tv'):
                loss = func['func'](pred)

            else:
                raise NotImplementedError(
                    f'loss_type {loss_type} is not implemented in Losser forward')

            loss *= func['weight']
            self.last_losses[func['loss_name']] = loss.item()

            total_loss += loss

        return total_loss

    def get_last_losses(self) -> dict:
        return self.last_losses.copy()


def get_loss_func(loss_name: str, loss_type: str, loss_params: dict, device) -> dict:
    option_criterion = loss_params.get('criterion')
    if loss_type == 'class':
        loss_func = custom_loss.get_criterion(
            option_criterion=option_criterion,
            device=device)

    elif loss_type == 'pixel':
        loss_func = custom_loss.get_criterion(
            option_criterion=option_criterion,
            device=device)

    elif loss_type == 'perceptual':
        loss_func = custom_loss.PerceptualLoss(
            loss_params=loss_params,
            device=device
        )

    elif loss_type == 'tv':
        loss_func = custom_loss.TVLoss(
            option_criterion=option_criterion
        )

    else:
        raise NotImplementedError(f'loss_type {loss_type} is not implemented')

    weight = loss_params['weight']

    if loss_name is None:
        loss_name = loss_type

    return {'loss_name': loss_name,
            'loss_type': loss_type,
            'weight': weight,
            'func': loss_func
            }
