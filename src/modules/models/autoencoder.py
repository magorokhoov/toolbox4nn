# https://github.com/magorokhoov
# 4jun2022sat
# Mikhail Gorokhov
# Coding custom NN toolbox

import os
import time
import argparse
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

    def train_step(self) -> None:
        # self.sample
        img_A, img_B = self.sample['img_A'], self.sample['img_B']

        if len(self.gpu_ids) != 0:
            img_A.cuda()
            img_B.cuda()

        img_pred = self.networks['autoencoder'](img_A)
        for losser_name in self.lossers:
            self.losses[losser_name] = self.lossers[losser_name](img_pred, img_B)

        # optimizers_zero_grad()
        # losses_backward()
        # optimizers_step()
        # schedulers_step()
