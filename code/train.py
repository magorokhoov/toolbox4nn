# https://github.com/magorokhoov
# 4jun2022sat
# Mikhail Gorokhov
# Coding custom NN toolbox

#import torch
#import torch.nn as nn
#import torch.nn.functional as F

import os
import argparse
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt


from tqdm import tqdm
from tqdm import tqdm_notebook

#from utilsss.utilsss import *
#from models.archs.my_simple_CLNN import my_simple_CLNN
#from data.dataset_CL_2folder import Dataset_CL_2folder



def fit():
    parser = argparse.ArgumentParser()
    parser.add_argument('-option', type=str, required=True, help='Path to options file.')
    args = parser.parse_args()
    print(args.option)

    torch.backends.cudnn.benchmark = True # True
    torch.backends.cudnn.deterministic = False # don't impact on rx570

    with open('options/test_option.yml', 'r') as file_option:
        options = yaml.safe_load(file_option)

    BATCH_SIZE = options['datasets'].get('batch_size', 1)
    EPOCHS = options['train'].get('epochs', 1)

    path_train1 = options['datasets'].get('path_train1')
    path_train2 = options['datasets'].get('path_train2')
    path_test1 = options['datasets'].get('path_test1')
    path_test2 = options['datasets'].get('path_test2')

    dataset_train = Dataset_CL_2folder(path_train1, path_train2)
    dataset_test = Dataset_CL_2folder(path_test1, path_test2)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_test  = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False)

    in_nc = options['arch'].get('in_nc', 3)
    mid_nc = options['arch'].get('mid_nc', 64)
    out_nc = options['arch'].get('out_nc', 2)

    model = my_simple_CLNN(in_nc, mid_nc, out_nc)
    model.print_num_parameters()

    model = model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    iters = 0
    loss_val = 0.0
    
    total_loss = 0
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader_train)
        for sample in pbar:
            img, label = sample['img'], sample['label']
            img = img.to('cuda')

            #label = F.one_hot(label, num_classes=10)
            label = label.to('cuda').float()
            #print(label)
            #print(len(img))
            predict = model(img)
            #print(predict, label)
            loss = loss_fn(predict, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss
            loss_val = 0.9*loss_val + 0.1*loss.item()
            pbar.set_description(f'loss: {round(loss_val, 5)}, iters: {iters}')
            
            iters += 1
        
        for g in optimizer.param_groups:
            g['lr'] /= 1.11
            lr = g['lr']
        print(f'loss: {total_loss/len(dataloader_train)//0.00001*0.00001}\tSet lr to: {lr}')

if __name__ == '__main__':
    fit()