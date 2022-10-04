import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml

from utils import utils


def get_scheduler(optimizer, option_scheduler: dict, total_iters: int):
    scheme = option_scheduler.get('scheme', None)
    if scheme is None:
        raise NotImplementedError(
            'Scheduler is None. Please, add to config file')

    scheme = scheme.lower()
    if scheme == 'linear':
        end_factor = option_scheduler.get('end_factor')

        scheduler = lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=end_factor,
            total_iters=total_iters,
            last_epoch=-1)
    elif scheme == 'multistep':
        milestones = option_scheduler.get('milestones')
        milestones_rel = option_scheduler.get('milestones_rel')

        if milestones_rel is not None:
            milestones = [int(stone * total_iters) for stone in milestones_rel]

        gamma = option_scheduler['gamma']

        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
            last_epoch=-1)
    else:
        raise NotImplementedError(
            f'Neural Network [{scheme}] is not recognized. networks.py doesn\'t know {[scheme]}')

    return scheduler
