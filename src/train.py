import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from modules.models import (classae_model, ae_model, classificator, srgan_model)
from utils import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-option', type=str, required=True, help='Path to options file.')
    args = parser.parse_args()
    print(args.option)
    option_path = args.option

    with open(option_path, 'r') as file_option:
        option = yaml.safe_load(file_option)

    # path_log_file = option['logger'].get('path_log_file')
    # logger = get_root_logger('base', root=path_log_file, phase='train', screen=True, tofile=True)
    seed = option.get('random_seed')
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    task = option.get('task').lower()
    if task in ('class', 'classification'):
        model = classificator.Classificator(option)
    elif task in ('ae', 'autoencoder'):
        model = ae_model.AutoEncoder(option)
    elif task == 'srgan':
        model = srgan_model.SRGAN_Model(option)
    elif task == 'classae':
        model = classae_model.ClassAE(option)
    else:
        raise NotImplementedError(f'Toolbox4nn don\'t know [{task}]')

    model.train()

if __name__ == '__main__':
    main()
