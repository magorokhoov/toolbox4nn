import torch
import torch.nn as nn
import torch.nn.functional as F

from . import blocks as blocks

class StupidD(nn.Module):
    def __init__(self, option_arch: dict) -> None:
        super().__init__()

        in_nc = option_arch['in_nc']
        mid_nc = option_arch['mid_nc']
        #out_nc = option_arch['out_nc']
        act_type = option_arch['act_type']
        norm_type = option_arch['norm_type']
        norm_groups = option_arch['norm_groups']

        pad3 = nn.ReflectionPad2d(3)
        act = blocks.ActivationLayer(act_type=act_type)        
        
        conv0 = nn.Conv2d(in_nc, mid_nc, 7, stride=2, padding=0) # 512 -> 256
        down1 = blocks.BlockCNA( # 256 -> 128
            mid_nc, 2*mid_nc, 7, stride=2,
            act_type=act_type, norm_groups=norm_groups, norm_type=norm_type
        )
        down2 = blocks.BlockCNA( # 128 -> 64
            2*mid_nc, 4*mid_nc, 7, stride=2,
            act_type=act_type, norm_groups=norm_groups, norm_type=norm_type
        )
        down3 = blocks.BlockCNA( # 64 -> 32
            4*mid_nc, 4*mid_nc, 5, stride=2,
            act_type=act_type, norm_groups=norm_groups, norm_type=norm_type
        )
        block3_2 = blocks.BlockCNA( # 64 -> 32
            4*mid_nc, 4*mid_nc, 5, stride=2,
            act_type=act_type, norm_groups=norm_groups, norm_type=norm_type
        )
        down4 = blocks.BlockCNA( # 32 -> 16
            4*mid_nc, 4*mid_nc, 5, stride=2,
            act_type=act_type, norm_groups=norm_groups, norm_type=norm_type
        )
        block4_2 = blocks.BlockCNA( # 64 -> 32
            4*mid_nc, 4*mid_nc, 5, stride=2,
            act_type=act_type, norm_groups=norm_groups, norm_type=norm_type
        )
        last = nn.Conv2d(4*mid_nc, 1, kernel_size=3, stride=1, padding=0)

        self.model = nn.Sequential(
            pad3, act, conv0, down1, down2,
            down3, block3_2, down4, block4_2, last
        )

    def forward(self, x):
        return self.model(x)