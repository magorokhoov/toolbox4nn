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


def get_scheduler(optimizer, option_scheduler:dict):
    name = option_scheduler.get('name', None)
    if name is None:
        raise NotImplementedError(f'Scheduler is None. Please, add to config file')

    name = name.lower()
    if name == 'linear'.lower():
        #start_factor = option_scheduler.get('start_factor')
        end_factor = option_scheduler.get('end_factor')
        total_iters = option_scheduler.get('total_iters')
        #last_epoch = option_scheduler.get('last_epoch')

        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=end_factor, total_iters=total_iters, last_epoch=-1)
    else:
        raise NotImplementedError(f'Neural Network [{name}] is not recognized. networks.py doesn\'t know {[name]}')
    

    return scheduler