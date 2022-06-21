import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np


class UpBlock(nn.Module):
    def __init__(self, in_nc, out_nc, up_type: str, factor=2, kernel_size=3) -> None:
        super(UpBlock, self).__init__()

        block = []
        padding = (kernel_size-1)//2
        if up_type == 'upscale':
            block += [nn.UpsamplingBilinear2d(scale_factor=factor)]
            block += [nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, padding=padding)]
        elif up_type == 'shuffle':
            block += [nn.PixelShuffle(factor)]
            block += [nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, padding=padding)]
        elif up_type == 'transpose':
            block += [nn.ConvTranspose2d(in_nc, out_nc, kernel_size=kernel_size, padding=padding)]
        else:
            raise NotImplementedError(f'up_type {up_type} is not implemented')

        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor):
        return self.block(x)
