import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils


class DatasetCL2folder(torch.utils.data.Dataset):
    def __init__(self, option_ds: dict):
        super(DatasetCL2folder, self).__init__()

        self.path_dir1 = option_ds.get('path_dir1')
        self.path_dir2 = option_ds.get('path_dir2')
        self.listdir1 = sorted(os.listdir(self.path_dir1))
        self.listdir2 = sorted(os.listdir(self.path_dir2))

    def __getitem__(self, index):
        id_class = 0
        if len(self.listdir1) <= index:
            id_class += 1
            index -= len(self.listdir1)

        path_img = ''
        label = None
        if id_class == 0:
            path_img = os.path.join(self.path_dir1, self.listdir1[index])
        elif id_class == 1:
            path_img = os.path.join(self.path_dir2, self.listdir2[index])
        else:
            print(f'Bad id_class, must be 0 or 1, not {id_class}')

        img = cv2.imread(path_img, cv2.IMREAD_COLOR)
        img = utils.get_scaled(img, (128, 128))
        img = utils.get_norm_img(img)

        label = np.zeros(2, dtype=np.float32)
        label[id_class] = 1.0

        tensor_img = utils.npimg2tensor(img)
        tensor_label = torch.from_numpy(label)

        return {'img': tensor_img, 'label': tensor_label}

    def __len__(self):
        return len(self.listdir1) + len(self.listdir2)
