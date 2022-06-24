import os
import random

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

        img_A = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # img = utils.get_scaled(img, (256, 256))
        img_A = utils.get_random_cropped(img_A, (256,256))
        img_A = utils.get_norm_img(img_A)
        
        h, w, c = img_A.shape
        img_B = img_A.copy()
        if random.random() <= 0.7:
            img_A += np.random.normal(0, 0.035, size=(h,w,c)).clip(-0.1, 0.1)
        img_A = img_A.clip(0,1.0)


        tensor_img_A = utils.npimg2tensor(img_A)
        tensor_img_B = utils.npimg2tensor(img_B)

        return {'img_A': tensor_img_A, 'img_B': tensor_img_B, 'img_A_path': img_path, 'img_B_path': img_path}

    def __len__(self):
        return len(self.listdir)
