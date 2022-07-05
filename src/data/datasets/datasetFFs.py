import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils


class DatasetFFs(torch.utils.data.Dataset):
    def __init__(self, option_ds: dict):
        super().__init__()

        self.path_dirs = option_ds.get('path_dirs')
        #print(self.path_dirs)
        self.listdirs = sorted(os.listdir(self.path_dirs))
        self.list_images = []
        self.boundaries = [0]
        self.num_classes = len(self.listdirs)
        #print(self.listdirs)
        for folder_name in self.listdirs:
            #print(folder_name)
            self.list_images += [sorted(os.listdir(os.path.join(self.path_dirs, folder_name)))]
            self.boundaries += [len(self.list_images[-1]) + self.boundaries[-1]]
        self.boundaries.pop(0)

    def __len__(self):
        return self.boundaries[-1]

    def __getitem__(self, index):
        i = index
        class_id = 0
        while index >= self.boundaries[class_id]: # linear search but i could use binary search
            i -= len(self.list_images[class_id])
            class_id += 1

        #print(f'class_id: {class_id} i: {i}')
        img_name = self.list_images[class_id][i]
        img_path = os.path.join(self.path_dirs, self.listdirs[class_id], img_name)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        img = utils.get_random_cropped(img, (256,256))
        img = utils.get_norm_img(img)
        img_B = img.copy()


        h, w, c = img.shape
        if random.random() <= 0.7:
            img += np.random.normal(0, 0.035, size=(h,w,c)).clip(-0.1, 0.1)
        img = img.clip(0,1.0)

        label = np.zeros(self.num_classes, dtype=np.float32)
        label[class_id] = 1.0

        tensor_img = utils.npimg2tensor(img)
        tensor_label = torch.from_numpy(label)

        return {'img': tensor_img, 'label': tensor_label, 'img_path': img_path}
