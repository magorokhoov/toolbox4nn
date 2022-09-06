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
        print(self.listdirs)
        self.list_images = []
        self.num_classes = len(self.listdirs)

        for folder_name in self.listdirs:

            self.list_images += [sorted(os.listdir(os.path.join(self.path_dirs, folder_name)))]

    def __len__(self):
        return 100*self.num_classes

    def __getitem__(self, index):
        class_id = index % self.num_classes
        i = random.randint(0, len(self.list_images[class_id])-1)

        #print(f'class_id: {class_id} i: {i}')
        img_name = self.list_images[class_id][i]
        img_path = os.path.join(self.path_dirs, self.listdirs[class_id], img_name)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        #img = utils.get_random_cropped(img, (512,512))
        try: 
            img = cv2.resize(img, (320,320), interpolation=cv2.INTER_LINEAR)
        except:
            print(f'error with {img_path}')
            img = np.random.randn((320,320,3), dtype=np.uint8) 
        img = utils.get_norm_img(img)


        h, w, c = img.shape
        if random.random() <= 0.5:
            img = cv2.flip(img, 1)

        #if random.random() <= 0.3:
        #    img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)

        #if random.random() <= 0.7:
        #    img += np.random.normal(0, 0.03, size=(h,w,c)).clip(-0.05, 0.05)
        img = img.clip(0,1.0)

        label = np.zeros(self.num_classes, dtype=np.float32)
        label[class_id] = 1.0

        tensor_img = utils.npimg2tensor(img)
        tensor_label = torch.from_numpy(label)

        #print(tensor_label)

        return {'img': tensor_img, 'label': tensor_label, 'img_path': img_path}
