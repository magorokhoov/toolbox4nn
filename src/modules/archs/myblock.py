# Программа GraveStonePhoto
# COPYRIGHT GOROKHOV MIKHAIL ANTONOVICH 2022
# КОПИРАЙТ ГОРОХОВ МИХАИЛ АНТОНОВИЧ 2022
# My Github: https://github.com/magorokhoov

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN

import torch.utils.checkpoint as cp

def get_factor(n):
        factors = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors.append(d)
                n //= d
            else:
                d += 1
        if n > 1:
            factors.append(n)
        factors.append(1)
        factors.reverse()
        factors = factors[::2]
        
        ans = 1
        for fact in factors:
            ans *= fact
        del factors
        return ans

class UpConvBlock(nn.Module):
    def __init__(self, dim, out_nc=None, kernel_size=3, num_groups=None, pad=nn.ReflectionPad2d, use_norm = True):
        super(UpConvBlock, self).__init__()

        if num_groups == None:
            num_groups = 4
        if out_nc == None:
            out_nc = dim

        self.norm = None
        if use_norm:
            self.norm = nn.GroupNorm(num_channels=out_nc, num_groups=num_groups, affine=True)

        self.act = nn.SiLU(inplace=True)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        
        if pad != None:
            p = 0
            self.pad = nn.ReflectionPad2d(kernel_size//2)
        else:
            p = kernel_size//2
            self.pad = None

        self.conv = nn.Conv2d(dim, out_nc, kernel_size=kernel_size, padding=p, bias=True)

    
    def forward(self, x):
        #model = [upsample] + block + [norm, act]

        out = self.upsample(x)
        if self.pad != None:
            out = self.pad(out)
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        if self.act:
            out = self.act(out)

        return out

# Group Depthwise Block
class GDWBlock(nn.Module):
    def __init__(self, in_nc, out_nc, act_layer = nn.LeakyReLU, stride=1, use_res=True, use_sn = False, res_multi = 1.0, pad=None, kernel_size=3,
        norm_type='group', attention='none'):
        super(GDWBlock, self).__init__()

        assert stride <= 2, 'stride of GDWBlock must be 1 or 2'

        if not use_res:
            res_multi = 1.0

        self.use_res = use_res
        self.res_multi = res_multi
        self.pad = pad
        attention = attention.lower()

        if act_layer == nn.LeakyReLU:
            self.act = act_layer(0.2, inplace=True)
        else:
            self.act = act_layer(inplace=True)

        self.res_conv = None
        self.res_avg = None
        if use_res:
            if stride == 2:
                self.res_avg = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            
            self.use_cat = 2*in_nc == out_nc
            if self.use_cat:
                out_nc = in_nc
            if not self.use_cat and in_nc != out_nc:
                self.res_conv = nn.Conv2d(in_nc, out_nc, kernel_size=1)

        p = 0
        if pad != None:
            self.pad = pad(kernel_size//2)
        else:
            p = kernel_size//2


        if in_nc > out_nc:
            mid_nc = in_nc
        else:
            mid_nc = out_nc


        # Norm

        self.norm = None
        if norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(mid_nc, affine=True)
        elif norm_type == 'group':
            self.norm = nn.GroupNorm(num_channels=mid_nc, num_groups=mid_nc//4, affine=True)


        # Just Convs

        self.conv0 = nn.Conv2d(in_nc, mid_nc, kernel_size=1, bias=True)
        self.conv1 = nn.Conv2d(mid_nc, mid_nc, kernel_size=kernel_size, groups=get_factor(mid_nc), padding=p, stride=stride, bias=True)
        # act
        self.conv2 = nn.Conv2d(mid_nc, out_nc, kernel_size=1)


        # Apply Spectral Norm if need

        if use_sn:
            self.conv0 = SN(self.conv0)
            self.conv1 = SN(self.conv1)
            self.conv2 = SN(self.conv2)
    
        # Attention
        if attention == 'se':
            attention_block = SELayer(out_nc, act_layer)
        elif attention == 'srm':
            attention_block = SRMLayer(out_nc) # SELayer(out_nc, act_layer)
        elif attention == 'scse':
            reduction = max(in_nc//64, 2)
            attention_block = ChannelSpatialSELayer(out_nc, reduction_ratio=reduction) # SCSEBlock(out_nc, act_layer=act_layer, reduction=4) #
        elif attention == 'cbam':
            attention_block = CBAM(out_nc)


        if attention != None or attention != 'none':
            self.attention_block = attention_block
        
    def forward_block(self, x):
        out = self.conv0(x)
        out = self.norm(out) # self.norm(out)
        out = self.act(out)
        
        out = self.conv1(out)
        out = self.act(out)

        out = self.conv2(out)

        return out

    def forward(self, x):

        y = self.res_multi*self.forward_block(x)

        if self.use_res:

            # Concatinate - 2*in_nc == out_nc
            if self.use_cat:
                if self.res_avg:
                    out = torch.cat((self.res_avg(x), y), axis=1)
                else:
                    out = torch.cat((x, y), axis=1)

            # in_nc != out_nc
            elif self.res_conv:
                if self.res_avg:
                    out = self.res_conv(self.res_avg(x)) + y
                else:
                    out = self.res_conv(x) + y

            # in_nc == out_nc
            else:
                if self.res_avg:
                    out = self.res_avg(x) + y
                else:
                    out = x + y
        else:
            out = y
        
        return out

# Group Depthwise Block
class GDWBlock(nn.Module):
    def __init__(self, in_nc, out_nc, act_layer = nn.LeakyReLU, use_res=True, use_sn = False, pad=nn.ReflectionPad2d, kernel_size=3,
        norm_type='group', attention='none'):
        super(GDWBlock, self).__init__()

        self.use_res = use_res
        #  res_multi = 1.0, #self.res_multi = res_multi

        if act_layer == nn.LeakyReLU:
            self.act = act_layer(0.2, inplace=True)
        else:
            self.act = act_layer(inplace=True)

        if in_nc > out_nc:
            mid_nc = in_nc
        else:
            mid_nc = out_nc

        p = 0
        if pad != None:
            pad = pad(kernel_size//2)
        else:
            p = kernel_size//2


        ### Norms ###
        norm = None
        self.norm = None
        if norm_type == 'instance':
            norm = nn.InstanceNorm2d(mid_nc, affine=True)
        elif norm_type == 'group':
            norm = nn.GroupNorm(num_channels=mid_nc, num_groups=mid_nc//4, affine=True)


        ### Just Convs ###
        conv0 = nn.Conv2d(in_nc, mid_nc, kernel_size=1, bias=True)
        conv1 = nn.Conv2d(mid_nc, mid_nc, kernel_size=kernel_size, groups=get_factor(mid_nc), padding=p, bias=True)
        #act
        conv2 = nn.Conv2d(mid_nc, out_nc, kernel_size=1)


        ### Apply Spectral Norm if need ###
        if use_sn:
            conv0 = SN(conv0)
            conv1 = SN(conv1)
            conv2 = SN(conv2)
    

        ### Attention ###
        self.attention_block = None
        if attention == 'se':
            attention_block = SELayer(out_nc, act_layer)
        elif attention == 'srm':
            attention_block = SRMLayer(out_nc) # SELayer(out_nc, act_layer)
        elif attention == 'scse':
            reduction = min(4, max(in_nc//64, 2))
            attention_block = ChannelSpatialSELayer(out_nc, reduction_ratio=reduction) # SCSEBlock(out_nc, act_layer=act_layer, reduction=4) #
        elif attention == 'cbam':
            attention_block = CBAM(out_nc)


        if attention != None and attention != 'none':
            self.attention_block = attention_block
        

        self.conv0 = conv0
        # self.norm = norm
        # act
        if pad != None:
            self.pad = pad
        else:
            self.pad = None
        
        self.conv1 = conv1
        self.conv2 = conv2

    def forward(self, x):
        mid = self.conv0(x)
        if self.norm:
            mid = self.norm(mid) # self.norm(mid)
        mid = self.act(mid)

        if self.pad:
            mid = self.pad(mid)

        mid = self.conv1(mid)
        mid = self.act(mid)
        mid = self.conv2(mid)
        
        if self.attention_block != None:
            mid = self.attention_block(mid)

        if self.use_res:
            out = x + mid
        else:
            out = mid
        
        return out

# Group Depthwise Truck
class GDWTruck(nn.Module):
    def __init__(self, channels, num_blocks = 3, kernel_size=3, act_layer = nn.SiLU, use_sn = False, attention=None, pad=None):
        super(GDWTruck, self).__init__()

        if attention == None or attention == 'none':
            attention = ['none']
        if act_layer == nn.LeakyReLU:
            self.last_act = act_layer(0.2, inplace=True)
        else:
            self.last_act = act_layer(inplace=True)

        blocks = []
        for i in range(num_blocks):
            blocks += [GDWBlock(channels, channels, kernel_size=kernel_size, act_layer=act_layer,
                                    use_sn=use_sn, attention=attention[i%len(attention)], pad=pad, 
                                    norm_type='group'
                                    )]

        self.last_conv = nn.Conv2d(2*channels, channels, kernel_size=1)
        self.last_norm = nn.GroupNorm(num_channels=channels, num_groups=channels//4, affine=True)

        if use_sn:
            self.last_conv = SN(self.last_conv) 

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        out = torch.cat((x, self.blocks(x)), axis=1)

        out = self.last_conv(out)
        out = self.last_norm(out)
        out = self.last_act(out)

        return out

#############################
 ##  _   _                ##
 ## | \ | |               ##
 ## |  \| | _____      __ ##
 ## | . ` |/ _ \ \ /\ / / ##
 ## | |\  |  __/\ V  V /  ##
 ## \_| \_/\___| \_/\_/   ##
#############################                

class CNACResBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, res_multi=1.0, use_sn=False, attention = None, pad=nn.ReflectionPad2d, use_norm=True):
        super(CNACResBlock, self).__init__()

        assert (dim%4) == 0, f'dim  must be mod by 4!!1! (your dim is {dim})'
        assert kernel_size%2 == 1, f'kernel size+1 must be mod by 2 (your kernel_size is {kernel_size})'

        if pad != None:
            p0 = 0
            p1 = 0
            self.pad0 = pad(kernel_size//2)
            self.pad1 = pad(kernel_size//2)
        else:
            p0 = kernel_size//2
            p1 = kernel_size//2
            self.pad0 = None
            self.pad1 = None

        self.res_multi = res_multi

        self.conv0 = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=p0, bias=True)
        self.norm0 = None if use_norm==False else nn.GroupNorm(num_channels=dim, num_groups=dim//2, affine=True)
        self.act = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=p1, bias=True)

        if use_sn:
            self.conv0 = SN(self.conv0)
            self.conv1 = SN(self.conv1)

        self.attention_block = None
        if attention != None and attention != 'none':
            self.attention_block = get_attention_block(dim, attention)


    def forward(self, x):
        if self.pad0:
            out = self.pad0(x)
            out = self.conv0(out)
        else:
            out = self.conv0(x)
        if self.norm0:
            out = self.norm0(out) # self.norm0(out)
        out = self.act(out)
        if self.pad1:
            out = self.pad1(out)
        out = self.conv1(out)

        if self.attention_block != None:
            out = self.attention_block(out)

        if self.res_multi != 1.0:
            return x + self.res_multi*out
        else:
            return x + out

class DWCResBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, res_multi=1.0, use_sn=False, attention = None, pad=nn.ReflectionPad2d, use_norm=True):
        super(DWCResBlock, self).__init__()

        assert (dim%4) == 0, f'dim  must be mod by 4!!1! (your dim is {dim})'
        assert kernel_size%2 == 1, f'kernel size must be mod by 2 (your kernel_size is {kernel_size})'

        if pad != None:
            p = 0
            self.pad = pad(kernel_size//2)
        else:
            p = kernel_size//2
            self.pad = None
        self.res_multi = res_multi

        num_groups = dim//8 if dim >= 8 else dim
        num_norm_groups = max(num_groups, dim//2) # in GN #TODO check what better dim//2 or dim//4

        self.conv0 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=True)
        self.norm0 = None if use_norm==False else nn.GroupNorm(num_channels=dim, num_groups=num_norm_groups, affine=True)
        self.act = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=num_groups, padding=p, bias=True)
        #self.norm1 = nn.GroupNorm(num_channels=dim, num_groups=num_norm_groups, affine=True)
        # act
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=True)

        if use_sn:
            conv0 = SN(self.conv0)
            conv1 = SN(self.conv1)
            conv2 = SN(self.conv2)

        self.attention_block = None
        if attention != None and attention != 'none':
            self.attention_block = get_attention_block(dim, attention)



    def forward(self, x):
        out = self.conv0(x)
        if self.norm0:
            out = self.norm0(out) # self.norm0(out)
        out = self.act(out)

        if self.pad:
            out = self.pad(out)
        out = self.conv1(out)
        out = self.act(out)

        out = self.conv2(out)
        
        if self.attention_block != None:
            out = self.attention_block(out)
            
        if self.res_multi != 1.0:
            return x + self.res_multi*out
        else:
            return x + out

class TruckCat(nn.Module):
    def __init__(self, dim, num_blocks, block_type='cnac', kernel_size=3, res_multi=1.0, use_sn=False, use_norm=True,
                attention = None, pad=nn.ReflectionPad2d, last_kernel_size=1):
        super(TruckCat, self).__init__()

        assert kernel_size%2 == 1, f'kernel_size must be odd. Your kernel_size is {kernel_size}'

        if block_type == 'cnac':
            block = CNACResBlock
        elif block_type == 'dw':
            block = DWCResBlock

        if attention == None:
            attention = [None]

        act = nn.SiLU(inplace=True)

        blocks1 = []

        for i in range(num_blocks):
            blocks1 += [block(dim, kernel_size=kernel_size, res_multi=res_multi, use_sn=use_sn, attention=attention[i%len(attention)], pad=pad, use_norm=use_norm)]
        self.blocks1 = nn.Sequential(*blocks1)

        self.pad = None
        if last_kernel_size > 1:
            self.pad = nn.ReflectionPad2d(last_kernel_size//2)
        self.last_conv = nn.Conv2d(2*dim, dim, kernel_size=last_kernel_size)
        self.last_norm = None if use_norm == False else nn.GroupNorm(num_channels=dim, num_groups=dim//4, affine=True)
        self.act = nn.SiLU(inplace=True)

        if use_sn:
            self.last_conv = SN(self.last_conv) 

    def forward(self, x):
        out = torch.cat((x, self.blocks1(x)), axis=1)

        if self.pad != None:
            out = self.pad(out)

        out = self.last_conv(out)

        if self.last_norm:
            out = self.last_norm(out) # self.norm0(out)
        out = self.act(out)

        return out

class TruckCat_V2(nn.Module):
    def __init__(self, dim, num_blocks, block_type='cnac', kernel_size=3, res_multi=1.0, use_sn=False, use_norm=True, attention = None, pad=nn.ReflectionPad2d):
        super(TruckCat_V2, self).__init__()

        assert kernel_size%2 == 1, f'kernel_size must be odd. Your kernel_size is {kernel_size}'

        if block_type == 'cnac':
            block = CNACResBlock
        elif block_type == 'dw':
            block = DWCResBlock

        if attention == None:
            attention = [None]

        blocks1 = []

        for i in range(num_blocks):
            blocks1 += [block(dim, kernel_size=kernel_size, res_multi=res_multi, use_sn=use_sn, attention=attention[i%len(attention)], pad=pad, use_norm=use_norm)]
        self.blocks1 = nn.Sequential(*blocks1)

        self.last_conv = nn.Conv2d(2*dim, dim, kernel_size=1)

    def forward(self, x):
        out = torch.cat((x, self.blocks1(x)), axis=1)
        out = self.last_conv(out)

        return out        

class ConvoyRes(nn.Module):
    def __init__(self, dim, num_blocks, num_trucks, block_type='cnac', kernel_size=3, res_multi=1.0, res_truck_multi = 1.0, use_sn=False,
                attention = None, pad=nn.ReflectionPad2d, last_kernel_size=1):
        super(ConvoyRes, self).__init__()

        self.res_truck_multi = res_truck_multi

        trucks = []
        for i in range(num_trucks):
            trucks += [TruckCat(dim, num_blocks=num_blocks, block_type=block_type, kernel_size=kernel_size, res_multi=res_multi,
                                use_sn=use_sn, attention=attention, pad=pad, last_kernel_size=last_kernel_size)]
        self.convoy = nn.Sequential(*trucks)


    def forward(self, x):
        return x + self.res_truck_multi*self.convoy(x)



#########################################################################
 ##    _____    __     __                    __   .__                  ##
 ##   /  _  \ _/  |_ _/  |_   ____    ____ _/  |_ |__|  ____    ____   ##
 ##  /  /_\  \\   __\\   __\_/ __ \  /    \\   __\|  | /  _ \  /    \  ##
 ## /    |    \|  |   |  |  \  ___/ |   |  \|  |  |  |(  <_> )|   |  \ ##
 ## \____|__  /|__|   |__|   \___  >|___|  /|__|  |__| \____/ |___|  / ##
 ##         \/                   \/      \/                        \/  ##
#########################################################################

# Программа GraveStonePhoto
# COPYRIGHT GOROKHOV MIKHAIL ANTONOVICH 2022
# КОПИРАЙТ ГОРОХОВ МИХАИЛ АНТОНОВИЧ 2022
# My Github: https://github.com/magorokhoov

def get_attention_block(dim, attention_name):
    ### Attention ###
    attention_block = None
    if attention_name == 'se':
        attention_block = SELayer(dim, act_layer)
    elif attention_name == 'srm':
        attention_block = SRMLayer(dim) # SELayer(dim, act_layer)
    elif attention_name == 'scse':
        reduction = min(4, max(dim//64, 2))
        attention_block = ChannelSpatialSELayer(dim, reduction_ratio=reduction)
    elif attention_name == 'cbam':
        attention_block = CBAM(dim)
    elif attention_name == 'stam':
        attention_block = StAMLayer(dim)
    else:
        raise Exception(f'There is no attention with name {attention_name}')

    return attention_block

class SELayer(nn.Module):
    def __init__(self, channel, act_layer, reduction=16):
        super(SELayer, self).__init__()

        if act_layer == nn.LeakyReLU:
            act = act_layer(0.2, inplace=True)
        else:
            act = act_layer(inplace=True)


        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        block = (nn.Linear(channel, channel // reduction, bias=True),
                           act, nn.Linear(channel // reduction, channel,
                           bias=True), nn.Sigmoid() )
        self.fc = nn.Sequential(*block)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SRMLayer(nn.Module):
    def __init__(self, channel, act_layer=None, reduction=None):
        # Reduction for compatibility with layer_block interface
        super(SRMLayer, self).__init__()

        # CFC: channel-wise fully connected layer
        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, bias=True,
                             groups=channel)
        #TODO num_groups: channel//4, channel//2 or just InstanceNorm?
        #self.norm = nn.GroupNorm(num_channels=channel, num_groups=channel//4, affine=True)

    def forward(self, x):
        b, c, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        #z = self.norm(z)
        g = torch.sigmoid(z)
        g = g.view(b, c, 1, 1)

        return x * g.expand_as(x)

class StAMLayer(nn.Module):
    def __init__(self, channel, act_layer=None, reduction=None):
        # Reduction and act_layer for compatibility with layer_block interface
        super(StAMLayer, self).__init__()

        # CFC: channel-wise fully connected layer
        self.cfc = nn.Conv1d(channel, channel, kernel_size=3, bias=True,
                             groups=channel)
        #TODO num_groups: channel//4, channel//2 or just InstanceNorm?
        #self.norm = nn.GroupNorm(num_channels=channel, num_groups=channel//4, affine=True)

    def forward(self, x):
        b, c, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        median = x.view(b, c, -1).median(-1).values.unsqueeze(-1)
        u = torch.cat((mean, std, median), -1)  # (b, c, 2)

        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        #z = self.norm(z)
        g = torch.sigmoid(z)
        g = g.view(b, c, 1, 1)

        return x * g.expand_as(x)

class SCSEBlock(nn.Module):
    def __init__(self, channel, act_layer = nn.LeakyReLU, reduction=16):
        super().__init__()
        if act_layer == nn.LeakyReLU:
            act = act_layer(0.2, inplace=True)
        else:
            act = act_layer(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                act,
                                                nn.Linear(int(channel // reduction), channel))
        self.spatial_se = nn.Conv2d(channel, 1, kernel_size=1,
                                    stride=1, padding=0, bias=True)

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = torch.sigmoid(self.channel_excitation(chn_se).view(bahs, chs, 1, 1))
        #print(f'x: {x}\t chn_se: {chn_se}')
        chn_se = x*chn_se
        #print(f'x: {x}\t chn_se: {chn_se}')
        spa_se = torch.sigmoid(self.spatial_se(x))
        spa_se = torch.mul(x, spa_se)
        #print(f'x: {x}\t chn_se: {spa_se}')
        return torch.add(chn_se, 1, spa_se)

class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        # print(input_tensor.size(), squeeze_tensor.size())
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        #output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor


class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018, arXiv:1803.02579*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, use_norm=True, bias=True):
        super(BasicConv, self).__init__()

        if (out_planes//4)*4 == out_planes:
            num_groups = out_planes//4
        elif (out_planes//2)*2 == out_planes:
            num_groups = out_planes//2
        else:
            num_groups = 1
        

        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.norm1 = nn.GroupNorm(num_channels=out_planes, num_groups=num_groups, eps=1e-5, affine=True) if use_norm else None
        self.act = nn.SiLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        if self.act is not None:
            x = self.act(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.SiLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

# Программа GraveStonePhoto
# COPYRIGHT GOROKHOV MIKHAIL ANTONOVICH 2022
# КОПИРАЙТ ГОРОХОВ МИХАИЛ АНТОНОВИЧ 2022
# My Github: https://github.com/magorokhoov