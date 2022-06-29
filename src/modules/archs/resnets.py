import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np

from . import blocks


class ResnetAE_v2(nn.Module):
    def __init__(self, option_arch: dict):
        super(ResnetAE_v2, self).__init__()

        in_nc = option_arch.get('in_nc')
        mid_nc = option_arch.get('mid_nc')
        out_nc = option_arch.get('out_nc')

        act_type = option_arch.get('act_type')
        norm_type = option_arch.get('norm_type')
        pad_type = option_arch.get('pad_type')
        up_type = option_arch.get('up_type')
        norm_groups = option_arch.get('norm_groups', 1)
        num_multiple = option_arch.get('num_blocks', 2)
        num_blocks = option_arch.get('num_multiple', 3)

        self.pad3 = nn.ReflectionPad2d(3)


        self.block1_1 = blocks.BlockCNA(in_nc, mid_nc, kernel_size=7, stride=2, pad_type='reflection', act_type=act_type, norm_type='none', norm_groups=norm_groups)
        self.block1_2 = blocks.BlockCNA(mid_nc, mid_nc, kernel_size=3, stride=1, pad_type=pad_type, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)  # 2x, 128

        self.resdown2 = blocks.ResDown(mid_nc, 2*mid_nc, kernel_size=7, stride=2, pad_type=pad_type, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)
        self.restruck2 = blocks.ResTruck(2*mid_nc, kernel_size=3, pad_type=pad_type, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups, num_blocks=num_blocks, num_multiple=num_multiple, residual_type='classic')
        self.resdown3 = blocks.ResDown(2*mid_nc, 4*mid_nc, kernel_size=7, stride=2, pad_type=pad_type, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)
        self.restruck3 = blocks.ResTruck(4*mid_nc, kernel_size=3, pad_type=pad_type, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups, num_blocks=num_blocks, num_multiple=num_multiple, residual_type='bottleneck')
        self.resdown4 = blocks.ResDown(4*mid_nc, 8*mid_nc, kernel_size=7, stride=2, pad_type=pad_type, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)
        self.restruck4 = blocks.ResTruck(8*mid_nc, kernel_size=3, pad_type=pad_type, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups, num_blocks=num_blocks, num_multiple=num_multiple, residual_type='bottleneck')
        
        self.up3 = blocks.UpBlock(2*mid_nc, 8*mid_nc, kernel_size=3, up_type=up_type, act_type=act_type) # 8x
        self.restruckup3 = blocks.ResTruck(8*mid_nc, kernel_size=3, pad_type=pad_type, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups, num_blocks=num_blocks, num_multiple=num_multiple, residual_type='bottleneck')
        self.up2 = blocks.UpBlock(2*mid_nc, 4*mid_nc, kernel_size=3, up_type=up_type, act_type=act_type) # 4x
        self.restruckup2 = blocks.ResTruck(4*mid_nc, kernel_size=3, pad_type=pad_type, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups, num_blocks=num_blocks, num_multiple=num_multiple, residual_type='bottleneck')
        self.up1 = blocks.UpBlock(1*mid_nc, 2*mid_nc, kernel_size=3, up_type=up_type, act_type=act_type) # 2xck
        self.restruckup1 = blocks.ResTruck(2*mid_nc, kernel_size=3, pad_type=pad_type, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups, num_blocks=num_blocks, num_multiple=num_multiple, residual_type='classic')
        self.up0 = blocks.UpBlock(mid_nc//2, mid_nc//2, kernel_size=3, up_type=up_type, act_type=act_type) # 1x
        self.blockup0_1 = blocks.BlockCNA(mid_nc//2, mid_nc//2, kernel_size=5, stride=1, pad_type=pad_type, act_type=act_type, norm_type='none', norm_groups=norm_groups)  # 1x, 256
 
        self.lastconv = nn.Conv2d(mid_nc//2, out_nc, kernel_size=7, padding=0)


    def forward(self, x):
        out = self.block1_1(x)
        out = self.block1_2(out)
        out = self.resdown2(out)
        out = self.restruck2(out)
        out = self.resdown3(out)
        out = self.restruck3(out)
        out = self.resdown4(out)
        out = self.restruck4(out)

        out = self.up3(out) # 8x
        out = self.restruckup3(out) # 8x
        out = self.up2(out) # 4x
        out = self.restruckup2(out) # 4x
        out = self.up1(out) # 2x
        out = self.restruckup1(out) # 2x
        out = self.up0(out) # 1x
        out = self.blockup0_1(out) # 1x
        
        out = self.pad3(out)
        out = self.lastconv(out)

        return out
