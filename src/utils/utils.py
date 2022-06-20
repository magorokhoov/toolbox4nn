import logging
import os
import re
import random
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def npimg2tensor(img):
    tensor = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(tensor)
    return tensor


def get_norm_img(img):
    return img.astype(np.float32) / 256.0 


def get_scaled(img, shape):
    img = cv2.resize(img, (shape[0], shape[1]), cv2.INTER_LINEAR)
    if len(img) == 2:
        np.expand_dims(img, axis=2)
    return img


def get_timestamp():
    return datetime.now().strftime('%Y%b%d%a-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_root_logger(
        logger_name=None,
        root=None,
        phase=None,
        level=logging.INFO,
        screen=False,
        tofile=True):
    """Set up logger. logger_name=None defaults to name 'base' """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return the base logger
    if not logger_name and logger.hasHandlers():
        return logger
    # else:
    #    print('lol')

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%d-%b-%Y-%a %H:%M:%S')
    logger.setLevel(level)

    if tofile:
        log_file = os.path.join(root, phase + f'_{get_timestamp()}.log')
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger


def sorted_nicely(l):
    def convert(text): return int(text) if text.isdigit() else text
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def dict2str(opt: dict, indent_l: int = 1) -> str:
    """Dictionary to string for logger."""
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def time_nicer(tm, point_digits=2) -> str:
    # tm must be in seconds
    if tm >= 60 * 60 * 2:
        tm /= 3600.0
        tm = f'{tm:.2f}h'
    elif tm >= 60 * 2:
        tm /= 60.0
        tm = f'{tm:.2f}m'
    else:
        tm = f'{tm:.2f}s'

    return tm
