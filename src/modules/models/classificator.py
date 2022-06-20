# https://github.com/magorokhoov
# 4jun2022sat
# Mikhail Gorokhov
# Coding custom NN toolbox

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

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

from utils import utils
import data

import modules.networks as networks
import modules.optimizers as optimizers
import modules.losses as losses


from modules.models import base_model


class Classificator(base_model.BaseModel):
    def __init__(self, option):
        super(Classificator, self).__init__(option)

    def train_step(self) -> None:
        # self.sample
        img, label = self.sample['img'], self.sample['label']

        if len(self.gpu_ids) != 0:
            img.cuda()
            label.cuda()

        pred = self.networks['classificator'](img)
        for losser_name in self.lossers:
            self.losses[losser_name] = self.lossers[losser_name](pred, label)

        # optimizers_zero_grad()
        # losses_backward()
        # optimizers_step()
        # schedulers_step()
