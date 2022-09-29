import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils

import data.processing.image as image
import data.processing.augmennt as augmennt
import data.processing.parser as parser

from data.processing.augmennt.functional import to_tensor


class DatasetHQ(torch.utils.data.Dataset):
    def __init__(self, option_ds: dict):
        super().__init__()

        self.path_dir = option_ds.get('path_dir')
        self.listdir = sorted(os.listdir(self.path_dir))

        option_processing = option_ds['processing']
        self.option_transformation = option_processing['transformation']
        self.option_augmentation = option_processing['augmentation']

        self.mode = option_processing['mode']
        self.loader = option_processing['loader']

        self.transformation = augmennt.Compose(
            parser.parse_transform_pipeline(self.option_transformation)
        )
        self.augmentation = augmennt.Compose(
            parser.parse_transform_pipeline(self.option_augmentation)
        )

    def __getitem__(self, index):
        img_path = os.path.join(self.path_dir, self.listdir[index])

        img = image.read_image(img_path, mode=self.mode, loader=self.loader)

        img_GT = self.transformation(img)
        img_LQ = self.augmentation(img_GT)

        img_GT = to_tensor(img_GT, bgr2rgb=False)
        img_LQ = to_tensor(img_LQ, bgr2rgb=False)

        return {'img_A': img_LQ,
                'img_B': img_GT,
                'img_A_path': img_path,
                'img_B_path': img_path}

    def __len__(self):
        return len(self.listdir)
