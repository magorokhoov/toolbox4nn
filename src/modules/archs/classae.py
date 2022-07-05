import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np

from . import blocks


class Encoder_001(nn.Module):
    def __init__(self, option_arch: dict):
        super().__init__()

        in_nc = option_arch.get('in_nc')
        mid_nc = option_arch.get('mid_nc')
        inner_nc = option_arch.get('inner_nc')
        act_type = option_arch.get('act_type')

        norm_type = option_arch.get('norm_type')
        norm_groups = option_arch.get('norm_groups')

        #pad3 = nn.ReflectionPad2d(3)
        block1_1 = blocks.BlockCNA(
            in_nc, mid_nc, 7, stride=2, groups=1, pad_type='reflection',
            act_type=act_type, norm_type='none')
        block1_2 = blocks.BlockCNA(
            mid_nc, mid_nc, 3, stride=1, groups=1, pad_type='zero',
            act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)
        down2 = blocks.ResDown(
            mid_nc, 2*mid_nc, 5, stride=2, pad_type='zero',
            act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)
        truck2_1 = blocks.ResTruck(
            2*mid_nc, 3, groups=mid_nc//4, num_multiple=1, num_blocks=4, pad_type='zero',
            act_type=act_type, norm_type=norm_type, norm_groups=norm_groups,
            resblock_type = 'classic')
        down3 = blocks.ResDown(
            2*mid_nc, 4*mid_nc, 3, stride=2, pad_type='zero',
            act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)
        truck3_1 = blocks.ResTruck(
            4*mid_nc, 3, groups=mid_nc//4, num_multiple=2, num_blocks=4, pad_type='zero',
            act_type=act_type, norm_type=norm_type, norm_groups=norm_groups,
            resblock_type = 'dw')
        down4 = blocks.ResDown(
            4*mid_nc, 8*mid_nc, 3, stride=2, pad_type='zero',
            act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)
        truck4_1 = blocks.ResTruck(
            8*mid_nc, 3, groups=mid_nc//4, num_multiple=2, num_blocks=4, pad_type='zero',
            act_type=act_type, norm_type=norm_type, norm_groups=norm_groups,
            resblock_type = 'dw')

        inner = blocks.BlockCNA(
            8*mid_nc, inner_nc, 3, groups=1, pad_type='zero',
            act_type=act_type, norm_type=norm_type, norm_groups=mid_nc//4)

        self.model = nn.Sequential(
            block1_1, block1_2,
            down2, truck2_1,
            down3, truck3_1, 
            down4, truck4_1,
            inner)

    def forward(self, x):
        return self.model(x)

class Decoder_001(nn.Module):
    def __init__(self, option_arch: dict):
        super().__init__()

        out_nc = option_arch.get('out_nc')
        mid_nc = option_arch.get('mid_nc')
        inner_nc = option_arch.get('inner_nc')
        act_type = option_arch.get('act_type')

        norm_type = option_arch.get('norm_type')
        norm_groups = option_arch.get('norm_groups')

        shuffle2x = nn.PixelShuffle(2)
        block3 = blocks.BlockCNA(
            inner_nc//4, 4*mid_nc, 3, pad_type='zero',
            act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)
        truck3_1 = blocks.ResTruck(
            4*mid_nc, 3, groups=mid_nc//4, num_multiple=2, num_blocks=4, pad_type='zero',
            act_type=act_type, norm_type=norm_type, norm_groups=norm_groups,
            resblock_type = 'dw')

        # shuffle2x
        block2 = blocks.BlockCNA(
            mid_nc, 4*mid_nc, 3, pad_type='zero',
            act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)
        truck2_1 = blocks.ResTruck(
            4*mid_nc, 3, groups=mid_nc//4, num_multiple=2, num_blocks=4, pad_type='zero',
            act_type=act_type, norm_type=norm_type, norm_groups=norm_groups,
            resblock_type = 'dw')

        # shuffle2x
        block1 = blocks.BlockCNA(
            mid_nc, 2*mid_nc, 3, pad_type='zero',
            act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)
        truck1_1 = blocks.ResTruck(
            2*mid_nc, 3, num_multiple=2, num_blocks=4, pad_type='zero',
            act_type=act_type, norm_type=norm_type, norm_groups=norm_groups,
            resblock_type = 'classic')

        # shuffle2x
        block0 = blocks.BlockCNA(
            mid_nc//2, mid_nc//2, 3, pad_type='zero',
            act_type=act_type, norm_type='none')

        pad3 = nn.ReflectionPad2d(3)
        lastconv0 = nn.Conv2d(mid_nc//2, out_nc, 7)


        self.model = nn.Sequential(
            shuffle2x, block3, truck3_1,
            shuffle2x, block2, truck2_1,
            shuffle2x, block1, truck1_1,
            shuffle2x, block0, pad3, lastconv0
        )

    def forward(self, x):
        return self.model(x)

class Class_001(nn.Module):
    def __init__(self, option_arch: dict) -> None:
        super().__init__()

        inner_nc = option_arch.get('inner_nc')
        midclass_nc = option_arch.get('midclass_nc')
        class_num = option_arch.get('class_num')
        act_type = option_arch.get('act_type')
        dropout_rate = option_arch.get('dropout_rate')

        norm_type = option_arch.get('norm_type')
        norm_groups = option_arch.get('norm_groups')

        act = blocks.ActivationLayer(act_type=act_type)

        # 16x16-> 7x7
        after_inner = blocks.BlockCNA(
            inner_nc, midclass_nc, 3, stride=2, pad_type='none',
            act_type=act_type, norm_type=norm_type, norm_groups=norm_groups)

        drop = nn.Dropout(dropout_rate, inplace=True)

        avg_pool = nn.AdaptiveAvgPool2d((1,1))
        flat = nn.Flatten()
        linear0 = nn.Linear(midclass_nc, midclass_nc)
        linear1 = nn.Linear(midclass_nc, int((midclass_nc*class_num)**0.5))
        linear2 = nn.Linear(int((midclass_nc*class_num)**0.5), class_num)

        self.model = nn.Sequential(
            after_inner, avg_pool, flat,
            linear0, drop, act,
            linear1, drop, act,
            linear2
        )

    def forward(self, x, last_act_type='none'):
        if last_act_type == 'sigmoid':
            return F.sigmoid(self.model(x))

        elif last_act_type == 'softmax':
            return F.softmax(self.model(x))

        elif last_act_type == 'none':
            return self.model(x)

        else:
            raise NotImplementedError(f'last_act_type {last_act_type} is not implemented')