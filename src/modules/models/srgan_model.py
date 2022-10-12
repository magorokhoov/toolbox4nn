# https://github.com/magorokhoov

import argparse
import math
import os
import time
import logging

import cv2
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import data
import modules.metrics as metrics
import modules.losser as losser
import modules.stats as stats
import modules.networks as networks
import modules.optimizers as optimizers
import modules.schedulers as schedulers
from utils import utils

from modules.models import base_model

class SRGAN_Model(base_model.BaseModel):
    def __init__(self, option: dict):
        super().__init__(option=option)

        # There must be list of selfs:
        '''
        self.networks, self.lossers, self.optimizers, self.schedulers
        self.dataloader, self.batch_size
        self.amp_scaler, self.device, self.cast
        self.acc_statses, self.display_freq, self.experiments_root
        self.logger, self.metrics, self.saving_freq
        '''

    def train(self) -> None:
        
        self.timer = utils.TrainTimer(n_iters=self.n_iters, print_freq=self.print_freq)
        self.batch_size = self.batch_sizes['dataset_1']  # Todo:recode it
        self.dataloader = self.dataloaders['dataset_1'] # Todo:recode it

        n_epochs = math.ceil(self.n_iters / len(self.dataloader))

        self.logger.info(f'Dataset has {len(self.dataloader.dataset)} images')
        self.logger.info(f'Dataloader has {len(self.dataloader)} mini-batches (batch_size={self.batch_size})')
        self.logger.info(f'Total epochs: {n_epochs}, iters: {self.n_iters}')

        self.logger.info('Neural network parameters: ')
        for network_name in self.networks:
            num_params = sum(p.numel() for p in self.networks[network_name].parameters())
            self.logger.info(f'{network_name}: {num_params:,}')

        i = 1
        self.logger.info('')
        self.logger.info('Training started!')
        try:
            for epoch in range(1, n_epochs + 1):
                for self.sample in self.dataloader:
                    img_A = self.sample['img_A'].to(self.device)
                    img_B = self.sample['img_B'].to(self.device)

                    # Generator
                    ### Enable grads for G, disable grads for D
                    for p in self.networks['gen'].parameters():
                        p.requires_grad = True
                    for p in self.networks['dis'].parameters():
                        p.requires_grad = False

                    with self.cast(enabled=self.use_amp):
                        gen_pred = self.networks['gen'](img_A)
                        dis_fake_pred = self.networks['dis'](gen_pred)
                        if self.option_lossers['gan']['relativistic']:
                            dis_real_pred = self.networks['dis'](img_B)
                        else:
                            dis_real_pred = None

                        loss_gen_image = self.lossers['gen'](gen_pred, img_B.detach())
                        loss_gen_gan = self.lossers['gan'](
                            real=dis_real_pred,
                            fake=dis_fake_pred,
                            phase='gen'
                        )
                        
                    loss_gen = loss_gen_image + loss_gen_gan

                    # Update G weights
                    self.optimizers['gen'].zero_grad()
                    self.optimizers['dis'].zero_grad()
                    self.amp_scaler.scale(loss_gen).backward()
                    self.amp_scaler.step(self.optimizers['gen'])
                    self.amp_scaler.update()
                    self.schedulers['gen'].step()

                    # Discriminator
                    ### Disable grads for G, enable grads for D
                    for p in self.networks['gen'].parameters():
                        p.requires_grad = False
                    for p in self.networks['dis'].parameters():
                        p.requires_grad = True

                    with self.cast(enabled=self.use_amp):
                        dis_fake_pred = self.networks['dis'](gen_pred.detach())
                        dis_real_pred = self.networks['dis'](img_B.detach())
                        loss_dis = self.lossers['gan'](
                            fake=dis_fake_pred,
                            real=dis_real_pred,
                            phase='dis'
                        ) 

                    # Update d weights 
                    self.optimizers['gen'].zero_grad()
                    self.optimizers['dis'].zero_grad()
                    self.amp_scaler.scale(loss_dis).backward()
                    self.amp_scaler.step(self.optimizers['dis'])
                    self.amp_scaler.update()
                    self.schedulers['dis'].step()

                    # Statistics
                    self.acc_statses['gen'].add_accumulation(
                        self.lossers['gen'].get_last_losses()
                    )
                    self.acc_statses['gan'].add_accumulation(
                        self.lossers['gan'].get_last_losses()
                    )

                    # Metrics
                    metrics_dict = self.metricer.calc_dict_metrics(
                        img_B.detach().cpu().float(),
                        gen_pred.detach().cpu().float()
                    )
                    self.acc_statses['metrics'].add_accumulation(metrics_dict)

                    # Display images
                    if i % self.display_freq == 0:
                        utils.display_images(
                            result_img_dir=self.result_img_dir,
                            iter=i, tensor_images_list=[
                            img_A.detach().cpu(),
                            gen_pred.detach().cpu(),
                            img_B.detach().cpu(),    
                        ])
                    
                    # Print info
                    if i % self.print_freq == 0:
                        self.logger_info_str(epoch=epoch, iter=i)

                    # Save models
                    if i % self.saving_freq == 0:
                        self.save_models(iteration=i)

                    # Next iter
                    i += 1
                    if i > self.n_iters:
                        break

        except KeyboardInterrupt:
            self.logger.info(
                f'Training interrupted. Latest models are saving at epoch: {epoch}, iter: {i} ')
        finally:
            self.save_models(iteration=i, is_last=True)
            self.logger.info('Training is ending...')

    def logger_info_str(self, epoch:int, iter:int):
        super().logger_info_str(epoch=epoch, iter=iter)

    def save_models(self, iteration=0, is_last: bool = False) -> None:
        super().save_models(iteration=iteration, is_last=is_last)