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


class BaseModel:
    def __init__(self, option: dict):
        self.name = option.get('name')

        # Getting options from option:dict
        option_ds = option.get('dataset')
        option_networks = option.get('networks')
        option_logger = option.get('logger')
        option_weights = option.get('weights')
        option_experiments = option.get('experiments')
        option_train = option.get('train')
        option_lossers = option.get('lossers')
        self.metrics_list = option.get('metrics')

        # Experiments
        self.saving_freq = option_experiments.get('saving_freq')
        self.experiments_root = option_experiments.get('root')
        utils.mkdir(self.experiments_root)
        self.experiment_dir_path = os.path.join(
            self.experiments_root, self.name)
        utils.mkdir(self.experiment_dir_path)

        # Logger
        self.logger = utils.get_root_logger(
            'base',
            root=self.experiment_dir_path,
            phase='train',
            screen=True,
            tofile=True)
        self.logger.info(utils.dict2str(option))
        self.logger.info('Training initialization...')
        self.print_freq = option_logger.get('print_freq')

        # Train
        self.option_train = option_train
        self.n_iters = option_train.get('n_iters')
        self.metrics = option_train.get('metrics')
        device = option.get('device')
        self.device = device

        # AMP : Automatic Mixed Perciption (auto float16/float32)
        self.use_amp = option.get('use_amp', False)
        self.cast = autocast
        self.amp_scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            self.logger.info('AMP enabled')
        else:
            self.logger.info('AMP disabled')

        # Dataset
        self.batch_size = option_ds.get('batch_size')
        dataset = data.create_dataset(option_ds)
        self.dataloader = data.create_dataloader(dataset, option_ds)

        # Networks, Optimizers and Schedulers
        self.networks = nn.ModuleDict()
        self.optimizers = {}
        self.schedulers = {}
        for network_name in option_networks:
            # Networks
            option_network = option_networks.get(network_name)
            self.networks[network_name] = networks.get_network(option_network, device=device)
            
            if option_weights is not None:
                option_network_weight = option_weights.get(network_name)
                if option_network_weight is not None:

                    strict = option_network_weight.get('strict', False)
                    network_path = option_network_weight.get('path', None)

                    if network_path is not None:
                        self.networks[network_name].load_state_dict(
                            torch.load(network_path),
                            strict=strict)
                        self.logger.info(f'[{network_name}] has been loaded from {network_path} (strict: {strict})')
                else:
                    self.logger.info(f'[{network_name}] will be trained from scratch')
            else:
                self.logger.info(f'All models will be trained from scratch')
            
            # Optimizator
            option_optimizer = option_network.get('optimizer')
            if option_optimizer == 'global':
                option_optimizer = option['optimizer']

            self.optimizers[network_name] = optimizers.get_optimizer(
                self.networks[network_name].parameters(), option_optimizer)

            # Scheduler
            option_scheduler = option_network.get('scheduler')
            if option_scheduler == 'global':
                option_scheduler = option['scheduler']

            self.schedulers[network_name] = schedulers.get_scheduler(
                optimizer=self.optimizers[network_name],
                option_scheduler=option_scheduler,
                total_iters=self.n_iters)

        ####### End Networks, Optimizers and Schedulers #######

        # Lossers and AccumulationStatses
        self.lossers = nn.ModuleDict()
        self.acc_stats = {}
        for losser_name in option_lossers:
            option_losser = option_lossers.get(losser_name)
            self.lossers[losser_name] = losser.get_losser(option_losser=option_losser, device=device)
            self.acc_stats[losser_name] = stats.AccumulationStats()

        self.acc_stats['metrics'] = stats.AccumulationStats()

        self.losses = nn.ModuleDict()

        # Display images
        self.display_freq = option_experiments.get('display_freq', None)

        if self.display_freq is not None:
            self.result_img_dir = os.path.join(self.experiment_dir_path, 'display_images')

            utils.mkdir(self.result_img_dir)

    def train(self) -> None:
        raise NotImplementedError('Do not use model_base. Please, use concrete pipeline instand')


    def logger_info_str(self, epoch:int, iter:int):
        loss_gen_stats_str = ''
        for acc_name in self.acc_stats:
            loss_gen_stats_str += f'{acc_name}: <{self.acc_stats[acc_name].get_str(reset=True)}> '

        dtet_info = self.timer.get_stats_str(iter=iter)
        info_str = f'<epoch: {epoch}, iter: {iter}, {dtet_info}> |'
        # Add to inf_str info about networks (currently only lr)
        for network_name in self.networks:
            lr = self.schedulers[network_name].get_last_lr()[-1]
            info_str += f' {network_name}_lr={lr:.3e} '
        info_str += '| '
        # Then lossers states
        
        info_str += f'{loss_gen_stats_str} '
        self.logger.info(info_str)

        #info_str = self.logger_info_networks()

    def save_models(self, iteration=0, is_last: bool = False) -> None:
        self.logger.info('Saving models...')
        self.models_dir_path = os.path.join(self.experiment_dir_path, 'models')

        if not os.path.isdir(self.models_dir_path):
            os.mkdir(self.models_dir_path)

        if is_last:
            prefix = ''
        else:
            prefix = f'{str(iteration)}_'

        for network_name in self.networks:
            self.network_path = os.path.join(
                self.models_dir_path, f'{prefix}{network_name}.pth')
            torch.save(
                self.networks[network_name].state_dict(), self.network_path)
