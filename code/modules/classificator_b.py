# https://github.com/magorokhoov
# 4jun2022sat
# Mikhail Gorokhov
# Coding custom NN toolbox

#import torch
#import torch.nn as nn
#import torch.nn.functional as F

import os
import time
import argparse
import yaml
import options
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt


from tqdm import tqdm
from tqdm import tqdm_notebook

import data
import modules.networks as networks
import modules.optimizers as optimizers
import modules.losses as losses

from utilsss.utilsss import *


class Classificator:
    def __init__(self, option):
        path_log_file = option['logger'].get('path_log_file')
        self.logger = get_root_logger('base', root=path_log_file, phase='train', screen=True, tofile=True)
        self.logger.info(dict2str(option))
        self.logger.info('Classificator initialization...')

        self.name = option.get('name')
        gpu_ids = option.get('gpu_ids')
        option_ds = option.get('dataset')
        option_archs = option.get('archs')
        #option_optimizer = option.get('optimizer')
        option_loss = option.get('loss')
        option_logger = option.get('logger')
        option_train = option.get('train')
        option_weights = option.get('weights', {})
        option_experiments = option.get('experiments')


        # Checkpoints
        self.experiments_root = option_experiments.get('root')
        self.checkpoint_freq = option_experiments.get('checkpoint_freq')

        if not os.path.isdir(self.experiments_root):
            os.mkdir(self.experiments_root)

        self.experiment_dir_path = os.path.join(self.experiments_root, self.name)

        if not os.path.isdir(self.experiment_dir_path):
            os.mkdir(self.experiment_dir_path)
        ### Checkpoints End

        

        self.print_freq = option_logger.get('print_freq')

        self.n_iters = option_train.get('n_iters')
        self.metrics = option_train.get('metrics')
        self.batch_size = option_ds.get('batch_size')
        self.gpu_ids = gpu_ids
        self.optimizers = {}
        self.schedulers = {}

        dataset = data.create_dataset(option_ds)
        self.dataloader = data.create_dataloader(dataset, option_ds, gpu_ids)

        for arch_name in option_archs:
            option_arch = option_archs.get(arch_name)
            self.networks[arch_name] = networks.get_network(option_arch)
        
            # Optimizator
            option_optimizer = option_arch.get('optimizer')

            if option_optimizer == 'global':
                option_optimizer = option.get('optimizer')

            self.optimizers[arch_name] = optimizers.get_optimizer(self.networks[arch_name].parameters(), option_optimizer)
            self.schedulers[arch_name] = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        
        # Loss
        self.losser = losses.ClassificatorLoss(option_loss)

        # Load Model
        if option_weights is not None:
            self.load_models(option_weights)

        if len(gpu_ids) != 0:
            self.networks_to_cuda()

    def train(self) -> None:
        time_start = time.time()
        loss_print_freq = 0.0
        time_print_freq = time.time()
        loss_print_freq_total = {}
        n_epochs = math.ceil(self.n_iters/len(self.dataloader))

        self.logger.info(f'Dataset has {len(self.dataloader.dataset)} images')
        self.logger.info(f'Dataloader has {len(self.dataloader)} mini-batches (batch_size={self.batch_size})')
        self.logger.info(f'Total epochs: {n_epochs}, iters: {self.n_iters}')
        self.logger.info(f'Neural network parameters: {self.neural_network.get_num_parameters():,}')

        i = 0
        self.logger.info('')
        self.logger.info('Training started!')
        for epoch in range(1, n_epochs+1):
            for sample in self.dataloader:
                i += 1
                if i > self.n_iters:
                    break

                self.optimizers_zero_grad()

                img, label = sample['img'], sample['label']
                if len(self.gpu_ids) != 0:
                    img.cuda()
                    label.cuda()

                pred = self.network['classificator'](img)
                self.loss = self.losser(pred, label)

                self.loss.backward()
                self.optimizer.step()

                # Estimated Time
                et = ((time.time() - time_start) * (self.n_iters - i) / i) 
                lr = self.scheduler.get_last_lr()[-1] # self.optimizer.param_groups[0]['lr']


                if i % self.print_freq == 0:
                    dtime_print_freq = time.time() - time_print_freq
                    time_print_freq = time.time()

                    time_string = f'DT={time_nicer(dtime_print_freq)}, ET={time_nicer(et)}'
                    self.logger.info(f'<epoch: {epoch}, iter: {i}, {time_string:s}, lr: {lr:.3e}> | {self.losser.get_losses_str(reset=True)}')

                if i % self.checkpoint_freq == 0:
                    self.make_checkpoint(i)
             
            self.schedulers_step()

        print(self.logger.info(f'Training Classificator is ending...'))

    def optimizers_zero_grad(self) -> None:
        for network_name in self.optimizers:
            self.optimizer[network_name].step()

    def optimizers_step(self) -> None:
        for network_name in self.optimizers:
            self.optimizers[network_name].step()

    def schedulers_step(self) -> None:
        for network_name in self.schedulers:
            self.schedulers[network_name].step()

    def networks_to_cuda(self) -> None:
        for network_name in self.networks:
            self.networks[network_name].cuda()

    def make_checkpoint(self, iteration=0) -> None:
        self.logger.info(f'Checkpoint. Saving models...')
        self.models_dir_path = os.path.join(self.experiment_dir_path, 'models')

        if not os.path.isdir(self.models_dir_path):
            os.mkdir(self.models_dir_path)

        self.neural_network_path = os.path.join(self.models_dir_path, str(iteration) + '_arch.pth')
        self.save_models(self.neural_network_path)

    def save_models(self, option) -> None:
        torch.save(self.networks.state_dict(), path)
    
    def load_models(self, option_weights:dict) -> None:
        print(option_weights)
        arch_path = option_weights.get('arch')
        print(arch_path)
        if arch_path is not None:
            self.neural_network.load_state_dict(torch.load(arch_path))
    
        









