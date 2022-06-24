from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np


class NoneLayer(nn.Module):
    """
    Forward and backward grads just pass this layer
    Used in NormLayer, PaddingLayer, ActivationLayer (etc) if x_type is none
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


class UpBlock(nn.Module):
    def __init__(self, in_nc, out_nc, up_type: str, factor=2, kernel_size=3) -> None:
        super(UpBlock, self).__init__()

        block = []
        padding = (kernel_size-1)//2
        if up_type == 'upscale':
            block += [nn.UpsamplingBilinear2d(scale_factor=factor)]
            block += [nn.Conv2d(in_nc, out_nc,
                                kernel_size=kernel_size, padding=padding)]
        elif up_type == 'shuffle':
            block += [nn.PixelShuffle(factor)]
            block += [nn.Conv2d(in_nc, out_nc,
                                kernel_size=kernel_size, padding=padding)]
        elif up_type == 'transpose':
            block += [nn.ConvTranspose2d(in_nc, out_nc,
                                         kernel_size=kernel_size, padding=padding)]
        else:
            raise NotImplementedError(f'up_type {up_type} is not implemented')

        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor):
        return self.block(x)


class NormLayer(nn.Module):
    def __init__(self, channels: int, norm_type: str, affine: bool = True, groups: int = 1) -> None:
        super(NormLayer, self).__init__()

        if norm_type == 'none':
            self.norm_layer = NoneLayer()
        elif norm_type == 'batch':
            self.norm_layer = nn.BatchNorm2d(channels, affine=affine)
        elif norm_type == 'instance':
            self.norm_layer = nn.InstanceNorm2d(channels, affine=affine)
        elif norm_type == 'layer':
            self.norm_layer = nn.GroupNorm(
                num_groups=1, num_channels=channels, affine=affine)
        elif norm_type == 'group':
            self.norm_layer = nn.GroupNorm(
                num_groups=groups, num_channels=channels, affine=affine)
        else:
            raise NotImplementedError(
                f'norm_type {norm_type} is not implemented')

    def forward(self, x):
        return self.norm_layer(x)


class PaddingLayer(nn.Module):
    def __init__(self, pad_type: str, pad=0) -> None:
        super(PaddingLayer, self).__init__()

        if pad_type == 'none':
            self.pad_layer = NoneLayer()
        elif pad_type == 'zero':
            self.pad_layer = nn.ZeroPad2d(padding=pad)
        elif pad_type == 'reflection':
            self.pad_layer = nn.ReflectionPad2d(padding=pad)
        elif pad_type == 'replication':
            self.pad_layer = nn.ReplicationPad2d(adding=pad)
        else:
            raise NotImplementedError(
                f'pad_type {pad_type} is not implemented')

    def forward(self, x):
        return self.pad_layer(x)


class ActivationLayer(nn.Module):
    def __init__(self, act_type: str, inplace=True, negative_slope=0.2) -> None:
        super(ActivationLayer, self).__init__()

        if act_type == 'none':
            self.act_layer = NoneLayer()
        elif act_type == 'sigmoid':
            self.act_layer = nn.Sigmoid()
        elif act_type == 'tanh':
            self.act_layer = nn.Tanh()
        elif act_type == 'relu':
            self.act_layer = nn.ReLU(inplace=inplace)
        elif act_type == 'lrelu':
            self.act_layer = nn.LeakyReLU(
                negative_slope=negative_slope, inplace=inplace)
        elif act_type == 'rrelu':
            # lower and upper used by default (1/8, 1/3) for simplication and maybe for my lazyness
            self.act_layer = nn.RReLU(inplace=inplace)
        elif act_type == 'gelu':
            self.act_layer = nn.GELU()
        elif act_type == 'elu':
            self.act_layer = nn.ELU()
        else:
            raise NotImplementedError(
                f'act_type {act_type} is not implemented')

    def forward(self, x):
        return self.act_layer(x)


class BlockCNA(nn.Module):
    def __init__(self,
                in_nc, out_nc, kernel_size, stride=1,
                pad_type='none', pad=0,
                act_type='relu',
                norm_type='none', affine=True, norm_groups=1) -> None:
        super(BlockCNA, self).__init__()

        self.pad = PaddingLayer(pad_type=pad_type, pad=pad)
        self.conv = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=0)
        self.norm = NormLayer(channels=out_nc, norm_type=norm_type, affine=affine, groups=norm_groups)
        self.act = ActivationLayer(act_type=act_type, inplace=True)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        out = self.norm(out)
        out = self.act(out)

        return out

    def forward_conv_only(self, x):
        out = self.pad(x)
        out = self.conv(out)

        return out
