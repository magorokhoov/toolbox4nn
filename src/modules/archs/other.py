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
        conv1 = nn.Conv2d(mid_nc, 2*mid_nc, kernel_size=5, stride=2, padding=2)
        norm1 = blocks.NormLayer(2*mid_nc, groups=mid_nc//2, norm_type='group')
        conv2 = nn.Conv2d(2*mid_nc, 4*mid_nc, kernel_size=5, stride=2, padding=2)
        norm2 = blocks.NormLayer(4*mid_nc, groups=mid_nc, norm_type='group')

        truck = blocks.ResTruck(
            4*mid_nc, kernel_size=3, num_multiple=4, num_blocks=2,
            act_type='gelu', norm_groups=mid_nc, pad_type='zero'
        )

        act = blocks.ActivationLayer(act_type='gelu')

        self.model = nn.Sequential(pad3, conv0, act, conv1, norm1, act, conv2, norm2, act, truck)

    def forward(self, x):
        return self.model(x)

class StupidDe(nn.Module):
    def __init__(self, option_arch: dict):
        super().__init__()

        out_nc = option_arch.get('out_nc')
        mid_nc = option_arch.get('mid_nc')

        pad3 = nn.ReflectionPad2d(3)

        block0 = blocks.UpBlock(
            4*mid_nc, 2*mid_nc, up_type='upscale',
            factor=2, kernel_size=5, act_type='gelu'
        )
        block1 = blocks.UpBlock(
            2*mid_nc, mid_nc, up_type='upscale',
            factor=2, kernel_size=5, act_type='gelu'
        )
        block2 = blocks.UpBlock(
            mid_nc, mid_nc//2, up_type='upscale',
            factor=2, kernel_size=5, act_type='gelu'
        )

        conv3 = nn.Conv2d(mid_nc//2, mid_nc//2, kernel_size=3, stride=1, padding=1)
        conv4 = nn.Conv2d(mid_nc//2, out_nc, kernel_size=7, stride=1, padding=0)

        act = blocks.ActivationLayer(act_type='gelu')

        self.model = nn.Sequential(
            block0, block1, block2,
            conv3, act, pad3, conv4)

    def forward(self, x):
        return self.model(x)