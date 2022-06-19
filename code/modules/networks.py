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

from utilsss.utilsss import *


def get_network(option_network: dict):
    name = option_network.get('name', None)
    if name is None:
        raise NotImplementedError(f'Neural Network is None. Please, add name to config file')

    name = name.lower()
    if name == 'my_simple_CLNN'.lower():
        from modules.archs.my_simple_CLNN import my_simple_CLNN as network
    else:
        raise NotImplementedError(f'Neural Network [{name}] is not recognized. networks.py doesn\'t know {[name]}')
    

    return network(option_network)