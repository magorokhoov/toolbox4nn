import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml
import shutil

from utils import utils


def get_network(option_network: dict, device='cpu'):
    arch = option_network.get('arch', None)
    if arch is None:
        raise NotImplementedError(
            'Neural Network is None. Please, add arch to config file')

    arch = arch.lower()
    if arch == 'mysimpleCLNN'.lower():
        from modules.archs.mysimpleCLNN import MySimpleCLNN as network
    elif arch == 'SimpleAEUnet'.lower():
        from modules.archs.simpleAEUnet import SimpleAEUnet as network # VggAEUnet
    elif arch == 'VggAEUnet'.lower():
        from modules.archs.vggs import VggAEUnet as network # 
    elif arch == 'VggAE'.lower():
        from modules.archs.vggs import VggAE as network #
    elif arch == 'ResnetAE_v2'.lower():
        from modules.archs.resnets import ResnetAE_v2 as network # 
    elif arch == 'stupid_en'.lower():
        from modules.archs.other import StupidEn as network # 
    elif arch == 'stupid_de'.lower():
        from modules.archs.other import StupidDe as network # 
    elif arch == 'classae_en'.lower():
        from modules.archs.classae import Encoder_001 as network # 
    elif arch == 'classae_de'.lower():
        from modules.archs.classae import Decoder_001 as network # 
    elif arch == 'classae_class'.lower():
        from modules.archs.classae import Class_001 as network #

    # Discriminators
    elif arch == 'stupid_discriminator':
        from modules.archs.discriminators import StupidD as network

    # efficientnet net
    elif arch == 'efficientnet_b0_features'.lower():
        from torchvision.models import efficientnet_b0
        network = efficientnet_b0(pretrained=True)
        return network.features.to(device=device)
    elif arch == 'efficientnet_b1_features'.lower():
        from torchvision.models import efficientnet_b1
        network = efficientnet_b1(pretrained=True)
        return network.features.to(device=device)

    # features - only for loss
    elif arch == 'vgg16':
        from modules.archs.feature import VGG16_fea as network
        weight_path = './modules/weights/vgg16_fea.pth'
        model = network()
        model.load_state_dict(torch.load(weight_path))
        return model.to(device=device)

    elif arch == 'vgg19':
        from modules.archs.feature import VGG19_fea as network
        weight_path = './modules/weights/vgg19_fea.pth'
        model = network()
        model.load_state_dict(torch.load(weight_path))
        return model.to(device=device)

    # Wrong arch name
    else:
        raise NotImplementedError(
            f'Neural Network [{arch}] is not recognized. networks.py doesn\'t know {[arch]}')

    return network(option_network).to(device=device)
