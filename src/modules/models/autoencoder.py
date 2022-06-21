# https://github.com/magorokhoov
# 4jun2022sat
# Mikhail Gorokhov
# Coding custom NN toolbox

import os
import time
import argparse
from matplotlib import image
import yaml
import math
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import data
import modules.networks as networks
import modules.optimizers as optimizers
import modules.losser as losser
from modules.models import base_model

from utils import utils


class AutoEncoder(base_model.BaseModel):
    def __init__(self, option):
        super(AutoEncoder, self).__init__(option)

        self.display_freq = option['experiments']['display_freq']
        self.result_img_dir = os.path.join(self.experiment_dir_path, 'display_images')

        utils.mkdir(self.result_img_dir)

    def train_step(self, iter: int=0) -> None:
        # self.sample
        img_A, img_B = self.sample['img_A'], self.sample['img_B']

        if len(self.gpu_ids) != 0:
            img_A = img_A.cuda()
            img_B = img_B.cuda()

        img_pred = self.networks['ae'](img_A)
        for losser_name in self.lossers:
            self.losses[losser_name] = self.lossers[losser_name](img_pred, img_B)

        if iter % self.display_freq == 0:
            self.display_images(iter=iter, images_list=[img_A.detach().cpu(), img_B.detach().cpu(), img_pred.detach().cpu()])

        # optimizers_zero_grad()
        # losses_backward()
        # optimizers_step()
        # schedulers_step()

    def display_images(self, iter: int, images_list: list):
        result_img = utils.tensor2npimg(images_list[0][0])

        for i in range(1, len(images_list)):
            img = utils.tensor2npimg(images_list[i][0])
            result_img = np.concatenate((result_img, img), axis=1) # result image become more wider

        result_img = (255.0*result_img).clip(0,255).astype(np.uint8)

        result_img_path = os.path.join(self.result_img_dir, f'{iter}.png')

        cv2.imwrite(result_img_path, result_img)
