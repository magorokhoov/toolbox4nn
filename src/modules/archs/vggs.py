import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np

from . import blocks


class VggAE(nn.Module):
    def __init__(self, option_arch: dict):
        super(VggAE, self).__init__()

        in_nc = option_arch.get('in_nc')
        mid_nc = option_arch.get('mid_nc')
        out_nc = option_arch.get('out_nc')

        act_type = option_arch.get('act_type')
        norm_type = option_arch.get('norm_type')
        pad_type = option_arch.get('pad_type')
        up_type = option_arch.get('up_type')
        norm_groups = option_arch.get('norm_groups', 1)

        self.pad3 = nn.ReflectionPad2d(3)

        self.block1_1 = blocks.BlockCNA(in_nc, mid_nc, kernel_size=7, stride=2, pad_type='reflection', pad=3, act_type=act_type, norm_type='none', norm_groups=norm_groups)
        self.block1_2 = blocks.BlockCNA(mid_nc, mid_nc, kernel_size=3, stride=1, pad_type=pad_type, pad=1, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)  # 2x, 128
        self.block2_1 = blocks.BlockCNA(mid_nc, 2*mid_nc, kernel_size=7, stride=2, pad_type=pad_type, pad=3, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)  # 4x, 64
        self.block2_2 = blocks.BlockCNA(2*mid_nc, 2*mid_nc, kernel_size=3, stride=1, pad_type=pad_type, pad=1, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)  # 4x, 64
        self.block3_1 = blocks.BlockCNA(2*mid_nc, 4*mid_nc, kernel_size=7, stride=2, pad_type=pad_type, pad=3, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)  # 8x, 32
        self.block3_2 = blocks.BlockCNA(4*mid_nc, 4*mid_nc, kernel_size=3, stride=1, pad_type=pad_type, pad=1, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)  # 8x, 32
        self.block3_3 = blocks.BlockCNA(4*mid_nc, 4*mid_nc, kernel_size=3, stride=1, pad_type=pad_type, pad=1, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)  # 8x, 32
        self.block4_1 = blocks.BlockCNA(4*mid_nc, 8*mid_nc, kernel_size=7, stride=2, pad_type=pad_type, pad=3, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)  # 16x, 16
        self.block4_2 = blocks.BlockCNA(8*mid_nc, 8*mid_nc, kernel_size=3, stride=1, pad_type=pad_type, pad=1, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)  # 16x, 16
        self.block4_3 = blocks.BlockCNA(8*mid_nc, 8*mid_nc, kernel_size=3, stride=1, pad_type=pad_type, pad=1, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)  # 16x, 16
        
        self.up3 = blocks.UpBlock(2*mid_nc, 8*mid_nc, kernel_size=5, up_type=up_type, act_type=act_type) # 8x
        self.blockup3_1 = blocks.BlockCNA(8*mid_nc, 8*mid_nc, kernel_size=3, stride=1, pad_type=pad_type, pad=1, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)  # 8x, 32
        self.blockup3_2 = blocks.BlockCNA(8*mid_nc, 8*mid_nc, kernel_size=3, stride=1, pad_type=pad_type, pad=1, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)  # 8x, 32
        self.up2 = blocks.UpBlock(2*mid_nc, 4*mid_nc, kernel_size=5, up_type=up_type, act_type=act_type) # 4x
        self.blockup2_1 = blocks.BlockCNA(4*mid_nc, 4*mid_nc, kernel_size=3, stride=1, pad_type=pad_type, pad=1, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)  # 4x, 64
        self.blockup2_2 = blocks.BlockCNA(4*mid_nc, 4*mid_nc, kernel_size=3, stride=1, pad_type=pad_type, pad=1, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)  # 4x, 64
        self.up1 = blocks.UpBlock(1*mid_nc, 2*mid_nc, kernel_size=5, up_type=up_type, act_type=act_type) # 2xck
        self.blockup1_1 = blocks.BlockCNA(2*mid_nc, 2*mid_nc, kernel_size=3, stride=1, pad_type=pad_type, pad=1, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)  # 2x, 128
        self.blockup1_2 = blocks.BlockCNA(2*mid_nc, 2*mid_nc, kernel_size=3, stride=1, pad_type=pad_type, pad=1, act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)  # 2x, 128
        self.up0 = blocks.UpBlock(mid_nc//2, mid_nc//2, kernel_size=5, up_type=up_type, act_type=act_type) # 1x
        self.blockup0_1 = blocks.BlockCNA(mid_nc//2, mid_nc//2, kernel_size=5, stride=1, pad_type=pad_type, pad=2, act_type=act_type, norm_type='none', norm_groups=norm_groups)  # 1x, 256
 
        self.lastconv = nn.Conv2d(mid_nc//2, out_nc, kernel_size=7, padding=0)


    def forward(self, x):
        out = self.block1_1(x)
        out = self.block1_2(out)
        out = self.block2_1(out)
        out = self.block2_2(out)
        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block3_3(out)
        out = self.block4_1(out)
        out = self.block4_2(out)
        out = self.block4_3(out)

        out = self.up3(out) # 8x
        out = self.blockup3_1(out) # 8x
        out = self.blockup3_2(out) # 8x
        out = self.up2(out) # 4x
        out = self.blockup2_1(out) # 4x
        out = self.blockup2_2(out) # 8x
        out = self.up1(out) # 2x
        out = self.blockup1_1(out) # 2x
        out = self.blockup1_2(out) # 8x
        out = self.up0(out) # 1x
        out = self.blockup0_1(out) # 1x
        
        out = self.pad3(out)
        out = self.lastconv(out)

        return out


class VggAEUnet(nn.Module):
    def __init__(self, option_arch: dict):
        super(VggAEUnet, self).__init__()

        in_nc = option_arch.get('in_nc')
        mid_nc = option_arch.get('mid_nc')
        out_nc = option_arch.get('out_nc')

        self.pad3 = nn.ReflectionPad2d(3)
        self.act = nn.GELU()
        self.conv1_1 = nn.Conv2d(in_nc, mid_nc, kernel_size=7, stride=2)  # 2x, 128
        self.conv1_2 = nn.Conv2d(mid_nc, mid_nc, kernel_size=3, stride=1, padding=1)  # 2x, 128
        self.conv2_1 = nn.Conv2d(mid_nc, 2*mid_nc, kernel_size=7, stride=2, padding=3)  # 4x, 64
        self.conv2_2 = nn.Conv2d(2*mid_nc, 2*mid_nc, kernel_size=3, stride=1, padding=1)  # 4x, 64
        self.conv3_1 = nn.Conv2d(2*mid_nc, 4*mid_nc, kernel_size=7, stride=2, padding=3)  # 8x, 32
        self.conv3_2 = nn.Conv2d(4*mid_nc, 4*mid_nc, kernel_size=3, stride=1, padding=1)  # 8x, 32
        self.conv3_3 = nn.Conv2d(4*mid_nc, 4*mid_nc, kernel_size=3, stride=1, padding=1)  # 8x, 32
        self.conv4_1 = nn.Conv2d(4*mid_nc, 8*mid_nc, kernel_size=7, stride=2, padding=3)  # 16x, 16
        self.conv4_2 = nn.Conv2d(8*mid_nc, 8*mid_nc, kernel_size=3, stride=1, padding=1)  # 16x, 16
        self.conv4_3 = nn.Conv2d(8*mid_nc, 8*mid_nc, kernel_size=3, stride=1, padding=1)  # 16x, 16
        #self.conv5_1 = nn.Conv2d(8*mid_nc, 16*mid_nc, kernel_size=7, stride=2, padding=3)  # 16x, 16
        #self.conv5_2 = nn.Conv2d(16*mid_nc, 16*mid_nc, kernel_size=3, stride=1, padding=1)  # 32x, 8
        #self.conv5_3 = nn.Conv2d(16*mid_nc, 16*mid_nc, kernel_size=3, stride=1, padding=1)  # 32x, 8

        #self.up4 = blocks.UpBlock(4*mid_nc, 16*mid_nc, up_type='shuffle') # 16x
        #self.convup4_1 = nn.Conv2d(16*mid_nc, 16*mid_nc, kernel_size=3, stride=1, padding=1)  # 16x, 16
        self.up3 = blocks.UpBlock(2*mid_nc, 8*mid_nc, up_type='shuffle') # 8x
        self.convup3_1 = nn.Conv2d(8*mid_nc, 8*mid_nc, kernel_size=3, stride=1, padding=1)  # 8x, 32
        self.up2 = blocks.UpBlock(2*mid_nc, 4*mid_nc, up_type='shuffle') # 4x
        self.convup2_1 = nn.Conv2d(4*mid_nc, 4*mid_nc, kernel_size=3, stride=1, padding=1)  # 4x, 64
        self.up1 = blocks.UpBlock(1*mid_nc, 2*mid_nc, up_type='shuffle') # 2x
        self.convup1_1 = nn.Conv2d(2*mid_nc, 2*mid_nc, kernel_size=3, stride=1, padding=1)  # 2x, 128
        self.up0 = blocks.UpBlock(mid_nc//2, mid_nc//2, up_type='shuffle') # 1x
        self.convup0_1 = nn.Conv2d(mid_nc//2, mid_nc//2, kernel_size=3, stride=1, padding=1)  # 1x, 256
 
        self.lastconv = nn.Conv2d(mid_nc//2, out_nc, kernel_size=7, padding=0)

        self.conv1_2_1x1 = nn.Conv2d(mid_nc, 2*mid_nc, kernel_size=1, stride=1)
        self.conv3_2_1x1 = nn.Conv2d(4*mid_nc, 8*mid_nc, kernel_size=1, stride=1)

    def forward(self, x, k:dict={'mid1': 1.0, 'mid2': 1.0}):
        out = self.pad3(x)
        out = self.conv1_1(out)
        out = self.act(out)
        out = self.conv1_2(out)
        mid1 = self.conv1_2_1x1(out) # 4x
        out = self.act(out)
        out = self.conv2_1(out)
        out = self.act(out)
        out = self.conv2_2(out)
        out = self.act(out)
        out = self.conv3_1(out)
        out = self.act(out)
        out = self.conv3_2(out)
        out = self.act(out)
        mid2 = self.conv3_2_1x1(out)  # 16x
        out = self.conv3_3(out)
        out = self.act(out)
        out = self.conv4_1(out)
        out = self.act(out)
        
        out = self.conv4_2(out)
        out = self.act(out)
        out = self.conv4_3(out)
        out = self.act(out)
        #out = self.conv5_1(out)
        #out = self.act(out)
        #out = self.conv5_2(out)
        #out = self.act(out)
        #out = self.conv5_3(out)
        #out = self.act(out)

        #out = self.up4(out) # 16x
        #out = self.act(out)
        #out = self.convup4_1(out) # 16x
        #out = self.act(out)
        out = self.up3(out) # 8x
        out = out + mid2*k['mid2']
        out = self.act(out)
        out = self.convup3_1(out) # 8x
        out = self.act(out)
        out = self.up2(out) # 4x
        out = self.act(out)
        
        out = self.convup2_1(out) # 4x
        out = self.act(out)
        out = self.up1(out) # 2x
        out = self.act(out)
        out = out + mid1*k['mid1']
        out = self.convup1_1(out) # 2x
        out = self.act(out)
        out = self.up0(out) # 1x
        out = self.act(out)
        out = self.convup0_1(out) # 1x
        out = self.act(out)

        out = self.pad3(out)
        out = self.lastconv(out)

        # self.conv2_1_1x1(mid1)

        return out

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())

