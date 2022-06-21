import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml

from tqdm import tqdm
from tqdm import tqdm_notebook

from utils import utils


def get_network(option_network: dict):
    arch = option_network.get('arch', None)
    if arch is None:
        raise NotImplementedError(
            'Neural Network is None. Please, add arch to config file')

    arch = arch.lower()
    if arch == 'mysimpleCLNN'.lower():
        from modules.archs.mysimpleCLNN import MySimpleCLNN as network
    else:
        raise NotImplementedError(
            f'Neural Network [{arch}] is not recognized. networks.py doesn\'t know {[arch]}')

    return network(option_network)
