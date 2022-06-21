import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils


class DatasetImg(torch.utils.data.Dataset):
    def __init__(self, option_ds: dict):
        super(DatasetImg, self).__init__()

        self.path_dir = option_ds.get('path_dir')
        self.listdir = sorted(os.listdir(self.path_dir))


    def __getitem__(self, index):
        img_path = os.path.join(self.path_dir, self.listdir[index])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        img = utils.get_scaled(img, (256, 256))
        img = utils.get_norm_img(img)


        tensor_img_A = utils.npimg2tensor(img)
        tensor_img_B = tensor_img_A

        return {'img_A': tensor_img_A, 'img_B': tensor_img_B, 'img_A_path': img_path, 'img_B_path': img_path}

    def __len__(self):
        return len(self.listdir)
