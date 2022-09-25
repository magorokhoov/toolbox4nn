import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np


class Metricer:
    def __init__(self, option_metrics:dict) -> None:
        self.metric_funcs = []
        #print(option_metrics)
        for metric_name in option_metrics:
            metric_type = option_metrics[metric_name]['metric_type']
            parameters = option_metrics[metric_name]
            func = get_metric_func(metric_name, metric_type, parameters)
            self.metric_funcs += [func]

    def calc_dict_metrics(self, tensor1, tensor2=None):

        result_dict = {}

        for func in self.metric_funcs:
            metric_name = func['metric_name']
            metric_type = func['metric_type']

            if metric_type == 'psnr':
                result_dict[metric_name] = func['func'](tensor1, tensor2)
            else:
                raise NotImplementedError(
                    f'metric_type [{metric_type}] is not implemented in calc_dict_metrics')

        return result_dict


def get_metric_func(metric_name: str, metric_type: str, parameters) -> dict:
    if metric_type == 'psnr':
        metric_func = psnr_torch

    else:
        raise NotImplementedError(f'metric_type {metric_type} is not implemented')

    if metric_name is None:
        metric_name = metric_type

    return {'metric_name': metric_name,
            'metric_type': metric_type,
            'parameters': parameters, 
            'func': metric_func
            }


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
        diff = img1.clip(0.0, pixel_max) - img2.clip(0.0, pixel_max)
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


def get_accuracy(tensor_pred:torch.Tensor, tensor_target:torch.Tensor):
    b, _ = tensor_pred.shape
    pred = torch.argmax(tensor_pred, dim=1)
    target = torch.argmax(tensor_target, dim=1) 
    count = 0
    for i in range(b):
        if pred[i] == target[i]:
            count += 1

    return count/b


def get_metrics_dict(
    metrics:dict,
    img1:torch.Tensor=None,
    img2:torch.Tensor=None,
    pred:torch.Tensor=None,
    target:torch.Tensor=None):

    metrics_dict = {}
    #print(metrics)
    print(metrics)
    for metric_name in metrics:
        metric_type = metrics[metric_name]
        if metric_type == 'psnr':
            metrics_dict['psnr'] = psnr_torch(img1=img1, img2=img2)
        elif metric_type in ('acc', 'accuracy'):
            metrics_dict['acc'] = get_accuracy(tensor_pred=pred, tensor_target=target)
        else:
            raise NotImplementedError(
                f'Metric "{metric_name}" with type [{metric_type}] is not implemented')

    return metrics_dict
