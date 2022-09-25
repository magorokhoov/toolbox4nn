import logging
import os
import re
import random
import time
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

def tensor2npimg(tensor):
    img = tensor.numpy()
    img = np.transpose(img, (1, 2, 0))
    return img


def get_norm_img(img):
    return img.astype(np.float32) / 256.0


def get_scaled(img, shape):
    img = cv2.resize(img, (shape[0], shape[1]), cv2.INTER_LINEAR)
    if len(img) == 2:
        np.expand_dims(img, axis=2)
    return img

def get_random_cropped(img, shape):
    h, w, _ = img.shape

    if h - shape[0] < 2:
        rnd_y = 0
    else:
        rnd_y = random.randint(0, h - shape[0])

    if h - shape[0] < 2:
        rnd_x = 0
    else:
        rnd_x = random.randint(0, w - shape[1])

    if rnd_y != 0:
        img = img[rnd_y:rnd_y+shape[0]]

    if rnd_x != 0:
        img = img[:, rnd_x:rnd_x+shape[1]]

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

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s', datefmt='%d-%b-%Y-%a %H:%M:%S')
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


def sorted_nicely(lst: list):
    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(lst, key=alphanum_key)


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


def time_nicer(tm:float, point_digits=2) -> str:
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


class TrainTimer:
    """
    Return:
    1. Delta Time between logger prints (DT)
    2. Estimated Time of ending (ET)
    """
    def __init__(self, n_iters: int, print_freq:int) -> None:
        self.n_iters = n_iters
        self.print_freq = print_freq
        self.time_last_print = time.time()

    def get_stats(self, iter:int) -> dict:
        # DT - delta time
        # ET - estimated time
        dt = time.time() - self.time_last_print
        et = dt * (self.n_iters - iter) / self.print_freq

        self.time_last_print = time.time()

        return {'DT': dt, 'ET': et}

    def get_stats_str(self, iter:int) -> str:
        stats = self.get_stats(iter=iter)
        dt, et = stats['DT'], stats['ET']

        str_result = f'DT={time_nicer(dt)}, ET={time_nicer(et)}'

        return str_result


def display_images(result_img_dir:str, iter: int, tensor_images_list: list, mode='rgb'):
        result_img = tensor2npimg(tensor_images_list[0][0])

        for i in range(1, len(tensor_images_list)):
            img = tensor2npimg(tensor_images_list[i][0])
            result_img = np.concatenate((result_img, img), axis=1) # result image will be more wider

        result_img = (255.0*result_img).clip(0,255).astype(np.uint8)

        result_img_path = os.path.join(result_img_dir, f'{iter}.png')

        if mode == 'rgb':
            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        elif mode == 'bgr':
            pass
        else:
            raise NotImplementedError(f'[{mode}] is not implemented. Use rgb or bgr')
        cv2.imwrite(result_img_path, result_img)