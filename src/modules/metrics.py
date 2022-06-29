import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np


def psnr_np(img1, img2, pixel_max:int=1.0, clip=True, shave=0, single=True):
    if clip:
        diff = img1.clip(0.0, pixel_max) - img2.clip(0.0, pixel_max)
    else:
        diff = img1 - img2

    if shave:
        diff = diff[shave:-shave, shave:-shave]

    if single:
        mse = np.mean(diff**2)
    else:
        raise NotImplementedError('I\'m still not implement for non single np psnr')
        mse = (diff**2).mean([-3, -2, -1])

    return 10 * np.log10(pixel_max ** 2  / mse)

def psnr_torch(img1:torch.Tensor, img2:torch.Tensor, pixel_max:int=1.0, clip=True, shave=0, single=True):
    if clip:
        diff = img1.clip(0.0, pixel_max) - img2.clamp(0.0, pixel_max)
    else:
        diff = img1 - img2

    if shave:
        diff = diff[..., shave:-shave, shave:-shave]


    if single:
        mse = torch.mean(diff**2)
    else:
        mse = diff.pow(2).mean([-3, -2, -1])


    max_val_tensor: torch.Tensor = torch.tensor(pixel_max).to(img1.device).to(img1.dtype)
    return 10 * torch.log10(max_val_tensor ** 2  / mse)