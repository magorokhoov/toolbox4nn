import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np

from . import blocks


class StupidEn(nn.Module):
    def __init__(self, option_arch: dict):
        super().__init__()

        in_nc = option_arch.get('in_nc')
        mid_nc = option_arch.get('mid_nc')

        pad3 = nn.ReflectionPad2d(3)
        conv0 = nn.Conv2d(in_nc, mid_nc, kernel_size=7, stride=2, padding=0)
        conv1 = nn.Conv2d(mid_nc, 2*mid_nc, kernel_size=7, stride=2, padding=3)
        conv2 = nn.Conv2d(2*mid_nc, 4*mid_nc, kernel_size=7, stride=2, padding=3)

        act = blocks.ActivationLayer(act_type='gelu')

        self.model = nn.Sequential(pad3, conv0, act, conv1, act, conv2, act)

    def forward(self, x):
        return self.model(x)

class StupidDe(nn.Module):
    def __init__(self, option_arch: dict):
        super().__init__()

        out_nc = option_arch.get('out_nc')
        mid_nc = option_arch.get('mid_nc')

        pad3 = nn.ReflectionPad2d(3)
        shuffle2x = nn.PixelShuffle(2)
        conv0 = nn.Conv2d(4*mid_nc, 8*mid_nc, kernel_size=7, stride=1, padding=3)
        conv1 = nn.Conv2d(2*mid_nc, 4*mid_nc, kernel_size=7, stride=1, padding=3)
        conv2 = nn.Conv2d(mid_nc, 2*mid_nc, kernel_size=7, stride=1, padding=3)
        conv3 = nn.Conv2d(mid_nc//2, out_nc, kernel_size=7, stride=1, padding=0)

        act = blocks.ActivationLayer(act_type='gelu')

        self.model = nn.Sequential(conv0, act, shuffle2x, conv1, act, shuffle2x, conv2, act, shuffle2x, pad3, conv3)

    def forward(self, x):
        return self.model(x)