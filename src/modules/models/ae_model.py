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

class AutoEncoder(base_model.BaseModel):
    def __init__(self, option: dict):
        super().__init__(option=option)

        # There must be list of selfs:
        '''
        self.networks, self.lossers, self.optimizers, self.schedulers
        self.dataloader, self.batch_size
        self.amp_scaler, self.device, self.cast
        self.acc_stats, self.display_freq, self.experiments_root
        self.logger, self.metrics, self.saving_freq
        '''

    def train(self) -> None:
        
        self.timer = utils.TrainTimer(n_iters=self.n_iters, print_freq=self.print_freq)

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

                    with self.cast(enabled=self.use_amp):
                        encoded_features = self.networks['en'](img_A)
                        img_pred = self.networks['de'](encoded_features)
                        loss_gen = self.lossers['l_gen'](img_pred, img_B)

                    last_losses_dict = self.lossers['l_gen'].get_last_losses()
                    self.acc_stats['l_gen'].add_accumulation(last_losses_dict)
                    
                    # Loss Generator Backward
                    self.optimizers['de'].zero_grad()
                    self.optimizers['en'].zero_grad()
                    
                    loss_gen.backward()
                    
                    self.amp_scaler.step(self.optimizers['de'])
                    self.amp_scaler.update()
                    self.amp_scaler.step(self.optimizers['en'])
                    self.amp_scaler.update()

                    self.schedulers['de'].step()
                    self.schedulers['en'].step()

                    # Metrics
                    metrics_dict = metrics.get_metrics_dict(
                        metrics=self.metrics_list,
                        img1=img_B.detach().cpu(),
                        img2=img_pred.detach().cpu()
                        )
                    self.acc_stats['metrics'].add_accumulation(metrics_dict)

                    # Display images
                    if i % self.display_freq == 0:
                        utils.display_images(
                            result_img_dir=self.result_img_dir,
                            iter=i, tensor_images_list=[
                            img_A.detach().cpu(),
                            img_B.detach().cpu(),
                            img_pred.detach().cpu()])

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