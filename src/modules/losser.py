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
    
    if losser_type is None:
        raise NotImplementedError(
            'losser_type is None. Please, add losser_type to config file')

    losser_type = losser_type.lower()
    if losser_type in ('class', 'image'):
        option_losses = option_losser['losses']
        losser = Losser(option_losses=option_losses, device=device)
    elif losser_type == 'gan':
        losser = LosserGAN(option_losser=option_losser, device=device)
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

            if loss_type in ('pixel', 'class', 'feature'):
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


class LosserGAN(nn.Module):
    def __init__(self, option_losser: dict, device) -> None:
        super().__init__()

        self.real_val = option_losser['real_val']
        self.fake_val = option_losser['fake_val']
        self.device = device
        self.last_losses = {

        }

        loss_params = option_losser['loss_func']
        loss_type = loss_params['loss_type']

        self.loss_func = get_loss_func(
            loss_name='gan_loss_func',
            loss_type=loss_type,
            loss_params=loss_params,
            device=device
        )

        self.weight_gen = float(option_losser['weight_gen'])
        self.weight_dis_fake = float(option_losser['weight_dis_fake'])
        self.weight_dis_real = float(option_losser['weight_dis_real'])

        self.is_relativistic = option_losser['relativistic']

    def forward(
        self,
        fake:torch.Tensor,
        real:torch.Tensor=None,
        phase:str=None):

        if phase == 'gen':
            label_real = torch.empty_like(fake, device=self.device).fill_(self.real_val)
            if self.is_relativistic:
                loss_d_fake = self.loss_func['func'](fake - real.mean(0, keepdim=True), label_real)
            else:
                loss_d_fake = self.loss_func['func'](fake, label_real)
            loss = self.loss_func['func'](fake, label_real)
            loss *= self.weight_gen

            self.last_losses['l_g'] = loss.item()

        elif phase == 'dis':
            label_fake = torch.empty_like(fake, device=self.device).fill_(self.fake_val)
            label_real = torch.empty_like(real, device=self.device).fill_(self.real_val)

            if self.is_relativistic:
                loss_d_real = self.loss_func['func'](real - fake.mean(0, keepdim=True), label_real)
                loss_d_fake = self.loss_func['func'](fake - real.mean(0, keepdim=True), label_fake)
            else:
                loss_d_real = self.loss_func['func'](real, label_real)
                loss_d_fake = self.loss_func['func'](fake, label_fake)
            
            loss_d_fake *= self.weight_dis_fake
            loss_d_real *= self.weight_dis_real

            loss = 0.5 * (loss_d_real + loss_d_fake)

            # yes. D_fake and D_real is not a loss
            self.last_losses['D_fake'] = fake.mean().item()
            self.last_losses['D_real'] = real.mean().item()

            self.last_losses['l_d_fake'] = loss_d_fake.item()
            self.last_losses['l_d_real'] = loss_d_real.item()

        else:
            raise NotImplementedError(f'phase {phase} must be "gen" or "dis"')

        

        return loss

    def get_last_losses(self) -> dict:
        return self.last_losses.copy()


def get_loss_func(
    loss_name: str,
    loss_type: str,
    loss_params: dict,
    device) -> dict:

    option_criterion = loss_params.get('criterion')
    if loss_type == 'class':
        loss_func = custom_loss.get_criterion(
            option_criterion=option_criterion,
            device=device)

    elif loss_type == 'gan':
        loss_func = custom_loss.get_criterion(
            option_criterion=option_criterion,
            device=device)

    elif loss_type == 'pixel':
        loss_func = custom_loss.get_criterion(
            option_criterion=option_criterion,
            device=device)

    elif loss_type == 'feature':
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
