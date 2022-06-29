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
import modules.networks as networks
import modules.optimizers as optimizers
import modules.schedulers as schedulers
from utils import utils


class BaseModel:
    def __init__(self, option: dict):
        self.name = option.get('name')
        gpu_ids = option.get('gpu_ids')
        option_ds = option.get('dataset')
        option_networks = option.get('networks')
        option_logger = option.get('logger')
        option_train = option.get('train')
        self.option_train = option_train
        option_weights = option.get('weights')
        option_experiments = option.get('experiments')

        self.checkpoint_freq = option_experiments.get('checkpoint_freq')
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

        # Logger end

        self.n_iters = option_train.get('n_iters')
        self.metrics = option_train.get('metrics')
        self.batch_size = option_ds.get('batch_size')
        self.gpu_ids = gpu_ids
        self.networks = {}
        self.optimizers = {}
        self.schedulers = {}
        self.lossers = {}
        self.losses = {}

        dataset = data.create_dataset(option_ds)
        self.dataloader = data.create_dataloader(dataset, option_ds, gpu_ids)

        self.use_amp = option.get('use_amp', False)
        self.cast = autocast
        self.amp_scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            self.logger.info('AMP enabled')
        else:
            self.logger.info('AMP disabled')

        for network_name in option_networks:
            option_network = option_networks.get(network_name)
            self.networks[network_name] = networks.get_network(option_network)

            # Loss
            # TODO: make losser getter
            option_loss = option_network.get('loss')
            if option_loss == 'global':
                option_loss = option['loss']

            losser_type = option_network.get('losser_type')
            self.lossers[network_name] = losser.get_losser(losser_type, option_loss=option_loss)

            # To CUDA if needed
            if len(gpu_ids) != 0:
                self.networks[network_name] = self.networks[network_name].cuda()
                self.lossers[network_name] = self.lossers[network_name].cuda()
                self.lossers[network_name].funcs_to_cuda()            

            # Optimizator
            option_optimizer = option_network.get('optimizer')
            if option_optimizer == 'global':
                option_optimizer = option['optimizer']

            self.optimizers[network_name] = optimizers.get_optimizer(
                self.networks[network_name].parameters(), option_optimizer)


            option_scheduler = option_network.get('scheduler')
            if option_scheduler == 'global':
                option_scheduler = option['scheduler']

            optimizer = self.optimizers[network_name]
            self.schedulers[network_name] = schedulers.get_scheduler(
                optimizer=optimizer,
                option_scheduler=option_scheduler,
                total_iters=self.n_iters)
            
            #option_scheduler.get('milestones_rel')



        # Load Model
        if option_weights is not None:
            self.load_models(option_weights)

        if len(gpu_ids) != 0:
            self.networks_to_cuda()
            self.lossers_to_cuda()

    def train_step(self) -> None:
        # self.sample
        # train_step()
        # optimizers zero_grad, step, losses backward, etc

        self.logger.error(
            'Please do not use base_model. Use concrete model instand')
        raise NotImplementedError(
            'Please do not use base_model. Use concrete model instand')

    def train(self) -> None:
        time_start = time.time()
        time_last_iter = time.time()
        n_epochs = math.ceil(self.n_iters / len(self.dataloader))

        self.logger.info(f'Dataset has {len(self.dataloader.dataset)} images')
        self.logger.info(
            f'Dataloader has {len(self.dataloader)} mini-batches (batch_size={self.batch_size})')
        self.logger.info(f'Total epochs: {n_epochs}, iters: {self.n_iters}')
        self.logger_info_networks_params()

        i = 1
        self.logger.info('')
        self.logger.info('Training started!')
        try:
            for epoch in range(1, n_epochs + 1):
                for self.sample in self.dataloader:

                    self.train_step(iter=i)        

                    if i % self.print_freq == 0:
                        # Estimated Time
                        et = ((time.time() - time_last_iter) / (self.print_freq) * (self.n_iters - i) )

                        dtime_print_freq = time.time() - time_last_iter
                        time_last_iter = time.time()

                        time_string = f'DT={utils.time_nicer(dtime_print_freq)}, ET={utils.time_nicer(et)}'
                        info_str = self.logger_info_networks()
                        self.logger.info(
                            f'<epoch: {epoch}, iter: {i}, {time_string:s}> | {info_str}')

                    if i % self.checkpoint_freq == 0:
                        self.save_models(iteration=i)

                    self.schedulers_step()

                    i += 1
                    if i > self.n_iters:
                        break

        except KeyboardInterrupt:
            self.logger.info(
                f'Training interrupted. Latest models are saving at epoch: {epoch}, iter: {i} ')
        finally:
            self.save_models(iteration=i, is_last=True)
            self.logger.info('Training is ending...')

    def logger_info_networks(self) -> str:
        info_str = ''
        for network_name in self.networks:
            lr = self.schedulers[network_name].get_last_lr()[-1]
            losses_str = self.lossers[network_name].get_losses_str(reset=True)
            info_str += f'{network_name}: <lr: {lr:.3e}; {losses_str}> '

        return info_str

    def logger_info_networks_params(self) -> None:
        self.logger.info('Neural network parameters: ')
        for network_name in self.networks:
            num_params = sum(p.numel() for p in self.networks[network_name].parameters())
            self.logger.info(f'{network_name}: {num_params:,}')

    def losses_backward(self) -> None:
        for network_name in self.losses:
            self.amp_scaler.scale(self.losses[network_name]).backward()

    def optimizers_zero_grad(self) -> None:
        for network_name in self.optimizers:
            self.optimizers[network_name].zero_grad()

    def optimizers_step(self) -> None:
        for network_name in self.optimizers:
            self.amp_scaler.step(self.optimizers[network_name])
            self.amp_scaler.update()

    def schedulers_step(self) -> None:
        for network_name in self.schedulers:
            self.schedulers[network_name].step()

    def networks_to_cuda(self) -> None:
        for network_name in self.networks:
            self.networks[network_name] = self.networks[network_name].cuda()

    def lossers_to_cuda(self) -> None:
        for losser_name in self.lossers:
            self.lossers[losser_name] = self.lossers[losser_name].to('cuda')
            self.lossers[losser_name].funcs_to_cuda()

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

    def load_models(self, option_weights: dict) -> None:
        for network_name in self.networks:
            network_path = option_weights.get(network_name)

            if network_path is not None:
                self.networks[network_name].load_state_dict(
                    torch.load(network_path),
                    strict=False)
        self.logger.info('All models have been loaded!')
