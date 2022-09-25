import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils

import data.processing.image as image
import data.processing.transforms as transforms


class DatasetHQ(torch.utils.data.Dataset):
    def __init__(self, option_ds: dict):
        super().__init__()

        self.path_dir = option_ds.get('path_dir')
        self.listdir = sorted(os.listdir(self.path_dir))

    def __getitem__(self, index):
        img_path = os.path.join(self.path_dir, self.listdir[index])

        img = image.read_image(img_path, mode='rgb', loader='cv')

        # Instead concrete use parser for augs and trans
        trans = transforms.Compose([
            transforms.ToTensor()
        ])
        augs  = transforms.Compose([
            transforms.RandomGaussianNoise(p=0.5, mean=0, var_limit=(5, 30))
        ])

        # !!! swap trans and augs
        img_GT = trans(img)

        img_LQ = augs(img)
        img_LQ = trans(img_LQ)

        return {'img_A': img_GT,
                'img_B': img_LQ,
                'img_A_path': img_path,
                'img_B_path': img_path}

    def __len__(self):
        return len(self.listdir)
