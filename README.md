# Toolbox4nn
Toolbox4nn is PyTorch framework for research and developing. Priority is CV (classification, super-resolution, stylization, restoration, GAN, etc) but i think you can use it for NLP also.

*In near future I will make awesome readme and docs*

### Implemented Tasks
1. Classification
2. Autoencoding

### Feature
1. Losses: pixel, tv, class
2. cpu and gpu
3. yaml config files
4. awesome logger (watch logger example below)
5. display images during training
6. DT (delta time) and ET (estimated time) in the logger

## How to start?

1. Config your yaml config file in options
2. `python train.py -option <your config>` 

## Project Structure

```
.
├── experiments -- here will be your result of training/tests (logs, model weights, val_images, etc) 
└── src -- main code
    ├── data -- datasets, dataloaders and data proccessing (tranforms and augmentation)
    ├── modules -- losses, models, optimizers, schedulers, metrics (not implemented now), etc
    ├── options -- YAML confings for flexible, powerful and easy configuration
    ├── sh -- sh files start if you use linux like me
    ├── train.py -- main file
    └── utils -- useful stuff like loggers, nicer_timer, dict2str, etc
```
## YAML Config example
```yaml
name: ae_21jun2022
task: autoencoder
gpu_ids: [0]
dataset:
  name: img
  path_dir: '../Datasets/cats_dogs/training_set/cats/'
  #path_dir2:  '../Datasets/cats_dogs/training_set/dogs/'
  #path_test1: '../Datasets/cats_dogs/test_set/cats/'
  #path_test2: '../Datasets/cats_dogs/test_set/dogs/'
  batch_size: 16
  shuffle: True
  num_workers: 4

networks:
  ae:
    arch: SimpleAEUnet
    in_nc: 3
    mid_nc: 32
    out_nc: 3

    losser_type: image
    loss:
      pixel:
        criterion: {criterion_type: l1}
        weight: 1.0
      tv:
        criterion: {gamma: 2}
        weight: 0.01

    optimizer:
      name: adam
      beta1: 0.9
      beta2: 0.999
      lr: 1e-3

    scheduler:
      #scheme: linear
      #end_factor: 0.2

      scheme: multistep
      #milestones: []
      milestones_rel: [0.2, 0.4, 0.6, 0.8] 
      gamma: 0.5


weights: 
  # ae: '../experiments/ae_21jun2022/models/ae.pth'

experiments:
  root: '../experiments/'
  checkpoint_freq: 1000
  display_freq: 50

train:
  n_iters: 6000
  
#loss:
#  func_type: CrossEntropyLoss
#  weight: 1.0
  #reduction: mean
  #pixel_criterion: l2
  #pixel_weight: 100.0

logger:
  print_freq: 100
  save_log_file: True
  #path_log_file: '../logs'
   
```

## Logger
```
[21-Jun-2022-Tue 23:34:27] INFO: Classificator initialization...
[21-Jun-2022-Tue 23:34:27] INFO: Dataset [DatasetImg - img] is created.
[21-Jun-2022-Tue 23:34:27] INFO: Dataset has 4000 images
[21-Jun-2022-Tue 23:34:27] INFO: Dataloader has 250 mini-batches (batch_size=16)
[21-Jun-2022-Tue 23:34:27] INFO: Total epochs: 24, iters: 6000
[21-Jun-2022-Tue 23:34:27] INFO: Neural network parameters: 
[21-Jun-2022-Tue 23:34:27] INFO: ae: 865,219
[21-Jun-2022-Tue 23:34:27] INFO: 
[21-Jun-2022-Tue 23:34:27] INFO: Training started!
[21-Jun-2022-Tue 23:35:47] INFO: <epoch: 1, iter: 100, DT=79.76s, ET=78.43m> | ae: <lr: 1.000e-03; pixel_l1: 8.8053e-02, tv_2: 8.7199e-06> 
[21-Jun-2022-Tue 23:36:51] INFO: <epoch: 1, iter: 200, DT=63.63s, ET=61.50m> | ae: <lr: 1.000e-03; pixel_l1: 4.8637e-02, tv_2: 1.0024e-05> 
[21-Jun-2022-Tue 23:37:55] INFO: <epoch: 2, iter: 300, DT=64.10s, ET=60.90m> | ae: <lr: 1.000e-03; pixel_l1: 3.3224e-02, tv_2: 1.3036e-05> 
[21-Jun-2022-Tue 23:38:59] INFO: <epoch: 2, iter: 400, DT=64.02s, ET=59.75m> | ae: <lr: 1.000e-03; pixel_l1: 2.9164e-02, tv_2: 1.6492e-05> 
[21-Jun-2022-Tue 23:40:03] INFO: <epoch: 2, iter: 500, DT=64.24s, ET=58.89m> | ae: <lr: 1.000e-03; pixel_l1: 2.9443e-02, tv_2: 1.9946e-05> 
[21-Jun-2022-Tue 23:41:07] INFO: <epoch: 3, iter: 600, DT=64.44s, ET=57.99m> | ae: <lr: 1.000e-03; pixel_l1: 2.1672e-02, tv_2: 2.1197e-05> 
[21-Jun-2022-Tue 23:42:11] INFO: <epoch: 3, iter: 700, DT=64.05s, ET=56.58m> | ae: <lr: 1.000e-03; pixel_l1: 2.1862e-02, tv_2: 2.1767e-05> 
[21-Jun-2022-Tue 23:43:16] INFO: <epoch: 4, iter: 800, DT=64.24s, ET=55.67m> | ae: <lr: 1.000e-03; pixel_l1: 2.1336e-02, tv_2: 2.3277e-05> 
[21-Jun-2022-Tue 23:44:20] INFO: <epoch: 4, iter: 900, DT=63.89s, ET=54.31m> | ae: <lr: 1.000e-03; pixel_l1: 2.1615e-02, tv_2: 2.4749e-05> 
[21-Jun-2022-Tue 23:45:24] INFO: <epoch: 4, iter: 1000, DT=64.00s, ET=53.33m> | ae: <lr: 1.000e-03; pixel_l1: 1.9092e-02, tv_2: 2.6496e-05> 
[21-Jun-2022-Tue 23:45:24] INFO: Checkpoint. Saving models...
[21-Jun-2022-Tue 23:46:28] INFO: <epoch: 5, iter: 1100, DT=64.66s, ET=52.80m> | ae: <lr: 1.000e-03; pixel_l1: 1.8832e-02, tv_2: 2.7449e-05>
```

## Last but not least

This project was heavily inspired by [traiNNer](https://github.com/victorca25/traiNNer) (developed by [victorca25](https://github.com/victorca25)). Try this one. They are also have a great Discord server "Enchance Everything". You will find the link in victorca25 rep. 
