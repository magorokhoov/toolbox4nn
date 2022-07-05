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


class UpBlock(nn.Module):
    def __init__(self, in_nc, out_nc, up_type: str, factor=2, kernel_size=3, act_type:str='gelu') -> None:
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

        block += [ActivationLayer(act_type=act_type)]
        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor):
        return self.block(x)


class BlockCNA(nn.Module):
    def __init__(self,
                in_nc, out_nc, kernel_size, stride=1, groups=1,
                pad_type='none',
                act_type='relu',
                norm_type='none', affine=True, norm_groups=1) -> None:
        super(BlockCNA, self).__init__()

        self.pad = PaddingLayer(pad_type=pad_type, pad=(kernel_size-1)//2)
        self.conv = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=0, groups=groups)
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


class ResDown(nn.Module):
    def __init__(self,
            in_nc, out_nc, kernel_size, stride=2,
            pad_type='none',
            act_type='relu',
            norm_type='none', affine=True, norm_groups=1) -> None:
        super(ResDown, self).__init__()

        self.residual = BlockCNA(
                in_nc, out_nc, kernel_size, stride=stride,
                pad_type=pad_type,
                norm_type=norm_type, affine=affine, norm_groups=norm_groups,
                act_type=act_type)

        self.conv1x1 = nn.Conv2d(in_nc, out_nc, kernel_size=1, stride=1)
        self.act = ActivationLayer(act_type=act_type)
        self.avgpool = nn.AvgPool2d(stride)

    def forward(self, x):

        return self.act(self.residual(x) + self.avgpool(self.conv1x1(x)))

class ResCNA(nn.Module):
    def __init__(self,
            mid_nc, kernel_size,
            num_multiple: int=1,
            pad_type='none',
            act_type='relu',
            norm_type='none', affine=True, norm_groups=1) -> None:
        super(ResCNA, self).__init__()
        # res types = 'add', 'cat', 'catadd'
        residual = []

        for _ in range(num_multiple-1):
            residual += [BlockCNA(
                in_nc=mid_nc, out_nc=mid_nc, kernel_size=kernel_size, stride=1,
                pad_type=pad_type,
                norm_type=norm_type, affine=affine, norm_groups=norm_groups,
                act_type=act_type)]

        residual += [BlockCNA(
                in_nc=mid_nc, out_nc=mid_nc, kernel_size=kernel_size, stride=1,
                pad_type=pad_type,
                norm_type=norm_type, affine=affine, norm_groups=norm_groups,
                act_type='none')]
        self.residual = nn.Sequential(*residual)

        self.act = ActivationLayer(act_type=act_type)

    def forward(self, x):

        return self.act(x + self.residual(x))


class ResBottleneck(nn.Module):
    def __init__(self,
        mid_nc, kernel_size, groups=1,
        num_multiple: int=1,
        pad_type='none',
        act_type='relu',
        norm_type='none', affine=True, norm_groups=1) -> None:

        super(ResBottleneck, self).__init__()

        residual = []
        
        for _ in range(num_multiple-1):
            residual += [BlockCNA(
                in_nc=mid_nc, out_nc=mid_nc, kernel_size=1,
                norm_type=norm_type, affine=affine, norm_groups=norm_groups,
                act_type=act_type
            )]

            residual += [BlockCNA(
                in_nc=mid_nc, out_nc=mid_nc, kernel_size=kernel_size, groups=groups,
                norm_type=norm_type, affine=affine, norm_groups=norm_groups,
                pad_type=pad_type,
                act_type=act_type
            )]

        residual += [BlockCNA(
            in_nc=mid_nc, out_nc=mid_nc, kernel_size=1,
            norm_type='none',
            act_type='none'
        )]

        self.residual = nn.Sequential(*residual)

        self.act = ActivationLayer(act_type=act_type)

    def forward(self, x):

        return self.act(x + self.residual(x))


class ResTruck(nn.Module):
    def __init__(self,
            mid_nc, kernel_size, groups=1,
            num_multiple: int=1, num_blocks: int=1,
            pad_type='none',
            act_type='relu',
            norm_type='none', affine=True, norm_groups=1,
            resblock_type='classic') -> None:
        super(ResTruck, self).__init__()
        # res types = 'add', 'cat', 'catadd'
        blocks = []

        if resblock_type == 'classic':
            for _ in range(num_blocks):
                blocks += [ResCNA(
                    mid_nc, kernel_size, pad_type=pad_type,
                    num_multiple=num_multiple,
                    norm_type=norm_type, affine=affine, norm_groups=norm_groups,
                    act_type=act_type
                )]

        elif resblock_type == 'dw':
            for _ in range(num_blocks):
                blocks += [ResBottleneck(
                    mid_nc, kernel_size, groups=groups, pad_type=pad_type,
                    num_multiple=num_multiple,
                    norm_type=norm_type, affine=affine, norm_groups=norm_groups,
                    act_type=act_type
                )]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):

        return self.blocks(x)
