import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm, tqdm_notebook

from modules.models import (classae_model, ae_model, classificator)
from utils import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-option', type=str, required=True, help='Path to options file.')
    args = parser.parse_args()
    print(args.option)
    option_path = args.option

    torch.backends.cudnn.benchmark = True  # True
    torch.backends.cudnn.deterministic = False  # don't impact on rx570

    with open(option_path, 'r') as file_option:
        option = yaml.safe_load(file_option)

    # path_log_file = option['logger'].get('path_log_file')
    # logger = get_root_logger('base', root=path_log_file, phase='train', screen=True, tofile=True)
    task = option.get('task')
    if task in ('class', 'classification'):
        model = classificator.Classificator(option)
    elif task in ('ae', 'autoencoder'):
        model = ae_model.AutoEncoder(option)
    elif task == 'classae':
        model = classae_model.ClassAE(option)
    else:
        raise NotImplementedError(f'Toolbox4nn don\'t know [{task}]')

    model.train()

if __name__ == '__main__':
    main()
