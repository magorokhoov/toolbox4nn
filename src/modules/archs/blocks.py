#####################################
 ##  ____  _            _         ##
 ## |  _ \| |          | |        ##
 ## | |_) | | ___   ___| | _____  ##
 ## |  _ <| |/ _ \ / __| |/ / __| ##
 ## | |_) | | (_) | (__|   <\__ \ ##
 ## |____/|_|\___/ \___|_|\_\___/ ##
#####################################
                               
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoneLayer(nn.Module):
    """
    Forward and backward grads just pass this layer
    Used in NormLayer, PaddingLayer, ActivationLayer (etc) if x_type is none
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class NormLayer(nn.Module):
    def __init__(self, channels: int, norm_type: str, affine: bool = True, groups: int = 1):
        super().__init__()

        if norm_type == 'none':
            self.norm_layer = NoneLayer()
        elif norm_type == 'batch':
            self.norm_layer = nn.BatchNorm2d(channels, affine=affine)
        elif norm_type == 'instance':
            self.norm_layer = nn.InstanceNorm2d(channels, affine=affine)
        elif norm_type == 'layer':
            self.norm_layer = nn.GroupNorm(num_groups=1, num_channels=channels, affine=affine)
        elif norm_type == 'group':
            self.norm_layer = nn.GroupNorm(num_groups=groups, num_channels=channels, affine=affine)
        else:
            raise NotImplementedError(
                f'norm_type {norm_type} is not implemented')

    def forward(self, x):
        return self.norm_layer(x)

class PaddingLayer(nn.Module):
    def __init__(self, pad_type: str, pad=0):
        super().__init__()

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
    def __init__(self, act_type: str, inplace=True):
        super().__init__()

        if act_type == 'none':
            self.act_layer = NoneLayer()
        elif act_type == 'sigmoid':
            self.act_layer = nn.Sigmoid()
        elif act_type == 'tanh':
            self.act_layer = nn.Tanh()
        elif act_type == 'relu':
            self.act_layer = nn.ReLU(inplace=inplace)
        elif act_type == 'lrelu':
            self.act_layer = nn.LeakyReLU(0.2, inplace=inplace)
        elif act_type == 'rrelu':
            # lower and upper used by default (1/8, 1/3) for simplication
            # and maybe for my lazyness
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
                in_nc, out_nc, kernel_size, stride=1, groups=1,
                pad_type='zero',
                act_type='relu',
                norm_type='none', affine=True, norm_groups=1):
        super().__init__()

        if norm_type == 'none':
            bias = affine
        else:
            bias = False

        self.pad = PaddingLayer(
            pad_type=pad_type,
            pad=(kernel_size-1)//2
        )
        self.conv = nn.Conv2d(
            in_nc, out_nc, kernel_size=kernel_size,
            stride=stride, padding=0, groups=groups, bias=bias
        )
        self.norm = NormLayer(
            channels=out_nc, norm_type=norm_type,
            affine=affine, groups=norm_groups
        )
        self.act = ActivationLayer(
            act_type=act_type,
            inplace=True
        )

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

class ResBlock(nn.Module):
    def __init__(self,
            nc, kernel_size,
            num_multiple: int=1,
            pad_type='zero',
            act_type='relu',
            norm_type='none', affine=True, norm_groups=1):
        super().__init__()
        # res types = 'add', 'cat', 'catadd'
        residual = []

        for _ in range(num_multiple-1):
            residual += [BlockCNA(
                in_nc=nc, out_nc=nc, kernel_size=kernel_size, stride=1,
                pad_type=pad_type,
                norm_type=norm_type, affine=affine, norm_groups=norm_groups,
                act_type=act_type)]

        residual += [BlockCNA(
                in_nc=nc, out_nc=nc, kernel_size=kernel_size, stride=1,
                pad_type=pad_type,
                norm_type=norm_type, affine=affine, norm_groups=norm_groups,
                act_type='none')]
        self.residual = nn.Sequential(*residual)

        self.act = ActivationLayer(act_type=act_type)

    def forward(self, x):
        return self.act(x + self.residual(x))


class ResDWBlock(nn.Module):
    def __init__(self,
        nc, kernel_size, groups=1,
        num_multiple: int=1,
        pad_type='zero',
        act_type='relu',
        norm_type='none', affine=True, norm_groups=1):

        super().__init__()

        residual = []
        
        for _ in range(num_multiple-1):
            residual += [BlockCNA(
                in_nc=nc, out_nc=nc, kernel_size=1, groups=1,
                norm_type=norm_type, affine=affine, norm_groups=norm_groups,
                act_type=act_type
            )]

            residual += [BlockCNA(
                in_nc=nc, out_nc=nc, kernel_size=kernel_size, groups=groups,
                norm_type=norm_type, affine=affine, norm_groups=norm_groups,
                pad_type=pad_type,
                act_type=act_type
            )]

        residual += [BlockCNA(
            in_nc=nc, out_nc=nc, kernel_size=1,
            norm_type=norm_type, affine=affine, norm_groups=norm_groups,
            act_type='none'
        )]

        self.residual = nn.Sequential(*residual)

        self.act = ActivationLayer(act_type=act_type)

    def forward(self, x):
        return self.act(x + self.residual(x))

class ResTruck(nn.Module):
    def __init__(self,
            nc, kernel_size, groups=1,
            num_multiple: int=1, num_blocks: int=1,
            pad_type='zero',
            act_type='relu',
            norm_type='none', affine=True, norm_groups=1,
            resblock_type='classic'):
        super(ResTruck, self).__init__()
        # res types = 'add', 'cat', 'catadd'
        blocks = []

        if resblock_type == 'classic':
            for _ in range(num_blocks):
                blocks += [ResBlock(
                    nc, kernel_size, pad_type=pad_type,
                    num_multiple=num_multiple,
                    norm_type=norm_type, affine=affine, norm_groups=norm_groups,
                    act_type=act_type
                )]

        elif resblock_type == 'dw':
            for _ in range(num_blocks):
                blocks += [ResDWBlock(
                    nc, kernel_size, groups=groups, pad_type=pad_type,
                    num_multiple=num_multiple,
                    norm_type=norm_type, affine=affine, norm_groups=norm_groups,
                    act_type=act_type
                )]

        self.truck = nn.Sequential(*blocks)

    def forward(self, x):
        return self.truck(x)
