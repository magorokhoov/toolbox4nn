import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np

from . import blocks


class SimpleAEUnet(nn.Module):
    def __init__(self, option_arch: dict):
        super(SimpleAEUnet, self).__init__()

        in_nc = option_arch.get('in_nc')
        mid_nc = option_arch.get('mid_nc')
        out_nc = option_arch.get('out_nc')

        self.pixelshuffle2x = nn.PixelShuffle(upscale_factor=2)

        self.pad3 = nn.ReflectionPad2d(3)
        self.act = nn.GELU()
        self.convdown_1 = nn.Conv2d(in_nc, 2 * mid_nc, kernel_size=7, stride=2)  # 2x, 64
        self.conv_en1_1 = nn.Conv2d(2 * mid_nc, 2 * mid_nc, kernel_size=3, stride=1, padding=1)  # 2x, 64
        self.convdown_2 = nn.Conv2d(2 * mid_nc, 4 * mid_nc, kernel_size=3, stride=2, padding=1)  # 4x, 32
        self.conv_en2_1 = nn.Conv2d(4 * mid_nc, 4 * mid_nc, kernel_size=3, stride=1, padding=1)  # 4x, 32
        self.convdown_3 = nn.Conv2d(4 * mid_nc, 8 * mid_nc, kernel_size=3, stride=2, padding=1)  # 8x, 16

        self.blockup_3 = blocks.UpBlock(2*mid_nc, 4*mid_nc, up_type='shuffle')
        self.conv_de2_1_1x1 = nn.Conv2d(4 * mid_nc, 4 * mid_nc, kernel_size=1)  # 4x, 32
        self.conv_de2_1 = nn.Conv2d(4 * mid_nc, 4 * mid_nc, kernel_size=3, stride=1, padding=1)  # 4x, 32
        self.blockup_2 = blocks.UpBlock(mid_nc, 2*mid_nc, up_type='shuffle')
        self.conv_de1_1_1x1 = nn.Conv2d(2 * mid_nc, 2 * mid_nc, kernel_size=1)  # 4x, 32
        self.conv_de1_1 = nn.Conv2d(2 * mid_nc, 2 * mid_nc, kernel_size=3, stride=1, padding=1)  # 4x, 32
        self.blockup_1 = blocks.UpBlock(mid_nc//2, mid_nc//2, up_type='shuffle')

        self.lastconv = nn.Conv2d(mid_nc//2, out_nc, kernel_size=7, padding=0)

    def forward(self, x):
        mid = self.pad3(x)
        mid = self.convdown_1(mid)
        mid = self.act(mid)
        mid = self.conv_en1_1(mid)
        mid = self.act(mid)
        mid1 = mid
        mid = self.convdown_2(mid)
        mid = self.act(mid)
        mid = self.conv_en2_1(mid)
        mid = self.act(mid)
        mid2 = mid
        mid = self.convdown_3(mid)
        mid = self.act(mid)

        # Decoder
        mid = self.blockup_3(mid)
        mid = self.act(mid)
        mid = mid + self.conv_de2_1_1x1(mid2)
        mid = self.conv_de2_1(mid)
        mid = self.act(mid)
        mid = self.blockup_2(mid)
        mid = self.act(mid)
        mid = mid + self.conv_de1_1_1x1(mid1)
        mid = self.conv_de1_1(mid)
        mid = self.act(mid)
        mid = self.blockup_1(mid)
        mid = self.act(mid)

        mid = self.pad3(mid)
        out = self.lastconv(mid)

        return out

    def print_num_parameters(self):
        num_params_features = sum(p.numel() for p in self.features.parameters())
        num_params_classficator = sum(p.numel() for p in self.classificator.parameters())

        print(f'Features params: {num_params_features}')
        print(f'Classificator params: {num_params_classficator}')

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())