import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils


class DatasetA2B(torch.utils.data.Dataset):
    def __init__(self, option_ds: dict):
        super(DatasetA2B, self).__init__()

        self.path_dir1 = option_ds.get('path_dir1')
        self.path_dir2 = option_ds.get('path_dir2')
        self.listdir1 = sorted(os.listdir(self.path_dir1))
        self.listdir2 = sorted(os.listdir(self.path_dir2))

        assert self.listdir1 == self.listdir2, 'Dataset dirs must have same images'

    def __getitem__(self, index):
        img_A_path = os.path.join(self.path_dir1, self.listdir1[index])
        img_B_path = os.path.join(self.path_dir2, self.listdir2[index])

        img_A = cv2.imread(img_A_path, cv2.IMREAD_COLOR)
        img_B = cv2.imread(img_A_path, cv2.IMREAD_COLOR)

        img_A = utils.get_scaled(img_A, (128, 128))
        img_A = utils.get_norm_img(img_A)

        img_B = utils.get_scaled(img_B, (128, 128))
        img_B = utils.get_norm_img(img_B)


        tensor_img_A = utils.npimg2tensor(img_A)
        tensor_img_B = utils.npimg2tensor(img_B)

        return {'img_A': tensor_img_A, 'img_B': tensor_img_B, 'img_A_path': img_A_path, 'img_B_path': img_B_path}

    def __len__(self):
        return len(self.listdir1)
