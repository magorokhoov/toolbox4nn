import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml

from tqdm import tqdm
from tqdm import tqdm_notebook

from utilsss.utilsss import *

def get_optimizer(model_params, option_optimizer: dict):
    name = option_optimizer.get('name', None)
    if name is None:
        raise NotImplementedError(f'Optimizer is None. Please, add to config file')
    name = name.lower()

    optimizer = None
    if name == 'sgd':
        lr = float(option_optimizer.get('lr'))
        momentum = float(option_optimizer.get('momentum', 0.0))
        weight_decay = float(option_optimizer.get('weight_decay', 0.0))
        nesterov = option_optimizer.get('nesterov', False)
        dampening = float(option_optimizer.get('dampening', 0.0))
        #eps = option_optimizer.get('eps', None)

        optimizer = torch.optim.SGD(model_params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov, dampening=dampening)

    elif name in ('adam', 'adamw'):
        lr = float(option_optimizer.get('lr'))
        beta1 = float(option_optimizer.get('beta1', 0.9))
        beta2 = float(option_optimizer.get('beta2', 0.999))
        weight_decay = float(option_optimizer.get('weight_decay', 0.0))
        amsgrad = option_optimizer.get('amsgrad', False)
        #eps = option_optimizer.get('eps', None)

        Class_optimizer = torch.optim.AdamW if name == 'adamw' else torch.optim.Adam
        optimizer = Class_optimizer(model_params, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay, amsgrad=amsgrad)
    else:
        raise NotImplementedError(f'Optimizer [{name}] is not recognized. optimizers.py doesn\'t know {[name]}')
    

    return optimizer

def get_scheduler():
    pass