# Toolbox4nn
Toolbox4nn is PyTorch framework for research and developing. Priority is CV (classification, super-resolution, stylization, restoration, GAN, etc) but i think you can use it for NLP also.

*In near future I will make awesome readme and docs*

Let's be in touch! Join to our [Discord server](https://discord.gg/wm2dbNYAQE)!

### Implemented Tasks
1. Classification
2. Autoencoding

### Features
1. Losses: pixel, tv, classification
2. Criterions: l1, l2 (mse), elastic
2. cpu and gpu
3. yaml config files
4. awesome logger (watch logger example below)
5. display images during training
6. DT (delta time) and ET (estimated time) in the logger
7. Autosave weights of models every *checkpoint_freq*
8. Stop and save weights with Ctrl+C (Linux terminal KeyboardInterrupt)
9. You can easy make your own model inheriting BaseModel
10. Or if you want you can make your own model task model-pipeline
11. And other things I haven't written here

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
[21-Jun-2022-Tue 23:47:32] INFO: <epoch: 5, iter: 1200, DT=64.13s, ET=51.31m> | ae: <lr: 1.000e-03; pixel_l1: 1.8802e-02, tv_2: 2.9278e-05> 
[21-Jun-2022-Tue 23:48:37] INFO: <epoch: 6, iter: 1300, DT=64.33s, ET=50.39m> | ae: <lr: 5.000e-04; pixel_l1: 1.2959e-02, tv_2: 3.0541e-05> 
[21-Jun-2022-Tue 23:49:41] INFO: <epoch: 6, iter: 1400, DT=63.90s, ET=48.99m> | ae: <lr: 5.000e-04; pixel_l1: 1.1420e-02, tv_2: 3.0068e-05> 
[21-Jun-2022-Tue 23:50:45] INFO: <epoch: 6, iter: 1500, DT=63.92s, ET=47.94m> | ae: <lr: 5.000e-04; pixel_l1: 1.2446e-02, tv_2: 3.0679e-05> 
[21-Jun-2022-Tue 23:51:49] INFO: <epoch: 7, iter: 1600, DT=64.26s, ET=47.12m> | ae: <lr: 5.000e-04; pixel_l1: 1.1366e-02, tv_2: 3.0927e-05> 
[21-Jun-2022-Tue 23:52:53] INFO: <epoch: 7, iter: 1700, DT=63.96s, ET=45.84m> | ae: <lr: 5.000e-04; pixel_l1: 1.1131e-02, tv_2: 3.3171e-05> 
[21-Jun-2022-Tue 23:53:57] INFO: <epoch: 8, iter: 1800, DT=64.24s, ET=44.97m> | ae: <lr: 5.000e-04; pixel_l1: 1.1163e-02, tv_2: 3.3162e-05> 
[21-Jun-2022-Tue 23:55:01] INFO: <epoch: 8, iter: 1900, DT=64.03s, ET=43.76m> | ae: <lr: 5.000e-04; pixel_l1: 1.2090e-02, tv_2: 3.4398e-05> 
[21-Jun-2022-Tue 23:56:05] INFO: <epoch: 8, iter: 2000, DT=63.93s, ET=42.62m> | ae: <lr: 5.000e-04; pixel_l1: 1.0722e-02, tv_2: 3.3957e-05> 
[21-Jun-2022-Tue 23:56:05] INFO: Checkpoint. Saving models...
[21-Jun-2022-Tue 23:57:09] INFO: <epoch: 9, iter: 2100, DT=64.40s, ET=41.86m> | ae: <lr: 5.000e-04; pixel_l1: 1.0286e-02, tv_2: 3.3630e-05> 
[21-Jun-2022-Tue 23:58:13] INFO: <epoch: 9, iter: 2200, DT=63.97s, ET=40.51m> | ae: <lr: 5.000e-04; pixel_l1: 1.0308e-02, tv_2: 3.6129e-05> 
[21-Jun-2022-Tue 23:59:18] INFO: <epoch: 10, iter: 2300, DT=64.39s, ET=39.71m> | ae: <lr: 5.000e-04; pixel_l1: 1.1025e-02, tv_2: 3.4890e-05> 
[22-Jun-2022-Wed 00:00:22] INFO: <epoch: 10, iter: 2400, DT=63.97s, ET=38.38m> | ae: <lr: 5.000e-04; pixel_l1: 9.8954e-03, tv_2: 3.5473e-05> 
[22-Jun-2022-Wed 00:01:26] INFO: <epoch: 10, iter: 2500, DT=63.92s, ET=37.29m> | ae: <lr: 2.500e-04; pixel_l1: 8.0148e-03, tv_2: 3.6333e-05> 
[22-Jun-2022-Wed 00:02:30] INFO: <epoch: 11, iter: 2600, DT=64.31s, ET=36.44m> | ae: <lr: 2.500e-04; pixel_l1: 7.5178e-03, tv_2: 3.6236e-05> 
[22-Jun-2022-Wed 00:03:34] INFO: <epoch: 11, iter: 2700, DT=64.01s, ET=35.20m> | ae: <lr: 2.500e-04; pixel_l1: 7.8983e-03, tv_2: 3.6868e-05> 
[22-Jun-2022-Wed 00:04:38] INFO: <epoch: 12, iter: 2800, DT=64.33s, ET=34.31m> | ae: <lr: 2.500e-04; pixel_l1: 7.4449e-03, tv_2: 3.6457e-05> 
[22-Jun-2022-Wed 00:05:42] INFO: <epoch: 12, iter: 2900, DT=63.98s, ET=33.05m> | ae: <lr: 2.500e-04; pixel_l1: 7.2856e-03, tv_2: 3.7766e-05> 
[22-Jun-2022-Wed 00:06:46] INFO: <epoch: 12, iter: 3000, DT=63.89s, ET=31.94m> | ae: <lr: 2.500e-04; pixel_l1: 7.1027e-03, tv_2: 3.6811e-05> 
[22-Jun-2022-Wed 00:06:46] INFO: Checkpoint. Saving models...
[22-Jun-2022-Wed 00:07:51] INFO: <epoch: 13, iter: 3100, DT=64.59s, ET=31.22m> | ae: <lr: 2.500e-04; pixel_l1: 7.2790e-03, tv_2: 3.7346e-05> 
[22-Jun-2022-Wed 00:08:55] INFO: <epoch: 13, iter: 3200, DT=64.30s, ET=30.01m> | ae: <lr: 2.500e-04; pixel_l1: 7.5900e-03, tv_2: 3.7853e-05> 
[22-Jun-2022-Wed 00:10:00] INFO: <epoch: 14, iter: 3300, DT=64.51s, ET=29.03m> | ae: <lr: 2.500e-04; pixel_l1: 7.6075e-03, tv_2: 3.7301e-05> 
[22-Jun-2022-Wed 00:11:04] INFO: <epoch: 14, iter: 3400, DT=63.95s, ET=27.71m> | ae: <lr: 2.500e-04; pixel_l1: 6.7003e-03, tv_2: 3.7568e-05> 
[22-Jun-2022-Wed 00:12:08] INFO: <epoch: 14, iter: 3500, DT=64.31s, ET=26.80m> | ae: <lr: 2.500e-04; pixel_l1: 6.6518e-03, tv_2: 3.9687e-05> 
[22-Jun-2022-Wed 00:13:12] INFO: <epoch: 15, iter: 3600, DT=64.39s, ET=25.76m> | ae: <lr: 2.500e-04; pixel_l1: 7.1028e-03, tv_2: 3.8541e-05> 
[22-Jun-2022-Wed 00:14:16] INFO: <epoch: 15, iter: 3700, DT=64.09s, ET=24.57m> | ae: <lr: 1.250e-04; pixel_l1: 6.0990e-03, tv_2: 3.8723e-05> 
[22-Jun-2022-Wed 00:15:21] INFO: <epoch: 16, iter: 3800, DT=64.69s, ET=23.72m> | ae: <lr: 1.250e-04; pixel_l1: 5.9947e-03, tv_2: 3.9227e-05> 
[22-Jun-2022-Wed 00:16:25] INFO: <epoch: 16, iter: 3900, DT=64.02s, ET=22.41m> | ae: <lr: 1.250e-04; pixel_l1: 5.8679e-03, tv_2: 3.9926e-05> 
[22-Jun-2022-Wed 00:17:29] INFO: <epoch: 16, iter: 4000, DT=63.99s, ET=21.33m> | ae: <lr: 1.250e-04; pixel_l1: 5.6052e-03, tv_2: 3.8499e-05> 
[22-Jun-2022-Wed 00:17:29] INFO: Checkpoint. Saving models...
[22-Jun-2022-Wed 00:18:33] INFO: <epoch: 17, iter: 4100, DT=64.43s, ET=20.40m> | ae: <lr: 1.250e-04; pixel_l1: 5.8502e-03, tv_2: 4.0123e-05> 
[22-Jun-2022-Wed 00:19:37] INFO: <epoch: 17, iter: 4200, DT=63.96s, ET=19.19m> | ae: <lr: 1.250e-04; pixel_l1: 5.7098e-03, tv_2: 3.8606e-05> 
[22-Jun-2022-Wed 00:20:42] INFO: <epoch: 18, iter: 4300, DT=64.52s, ET=18.28m> | ae: <lr: 1.250e-04; pixel_l1: 5.7968e-03, tv_2: 3.9264e-05> 
[22-Jun-2022-Wed 00:21:46] INFO: <epoch: 18, iter: 4400, DT=63.85s, ET=17.03m> | ae: <lr: 1.250e-04; pixel_l1: 5.4013e-03, tv_2: 3.9180e-05> 
[22-Jun-2022-Wed 00:22:50] INFO: <epoch: 18, iter: 4500, DT=63.95s, ET=15.99m> | ae: <lr: 1.250e-04; pixel_l1: 5.6241e-03, tv_2: 4.0535e-05> 
[22-Jun-2022-Wed 00:23:54] INFO: <epoch: 19, iter: 4600, DT=64.22s, ET=14.98m> | ae: <lr: 1.250e-04; pixel_l1: 5.5777e-03, tv_2: 3.8569e-05> 
[22-Jun-2022-Wed 00:24:58] INFO: <epoch: 19, iter: 4700, DT=63.93s, ET=13.85m> | ae: <lr: 1.250e-04; pixel_l1: 5.4241e-03, tv_2: 4.0222e-05> 
[22-Jun-2022-Wed 00:26:02] INFO: <epoch: 20, iter: 4800, DT=64.61s, ET=12.92m> | ae: <lr: 1.250e-04; pixel_l1: 5.3837e-03, tv_2: 4.1271e-05> 
[22-Jun-2022-Wed 00:27:07] INFO: <epoch: 20, iter: 4900, DT=64.17s, ET=11.76m> | ae: <lr: 6.250e-05; pixel_l1: 5.1116e-03, tv_2: 4.0486e-05> 
[22-Jun-2022-Wed 00:28:11] INFO: <epoch: 20, iter: 5000, DT=64.32s, ET=10.72m> | ae: <lr: 6.250e-05; pixel_l1: 5.0025e-03, tv_2: 3.9635e-05> 
[22-Jun-2022-Wed 00:28:11] INFO: Checkpoint. Saving models...
[22-Jun-2022-Wed 00:29:15] INFO: <epoch: 21, iter: 5100, DT=64.51s, ET=9.68m> | ae: <lr: 6.250e-05; pixel_l1: 5.0082e-03, tv_2: 4.0487e-05> 
[22-Jun-2022-Wed 00:30:19] INFO: <epoch: 21, iter: 5200, DT=64.00s, ET=8.53m> | ae: <lr: 6.250e-05; pixel_l1: 4.9621e-03, tv_2: 4.0048e-05> 
[22-Jun-2022-Wed 00:31:24] INFO: <epoch: 22, iter: 5300, DT=64.30s, ET=7.50m> | ae: <lr: 6.250e-05; pixel_l1: 4.9881e-03, tv_2: 4.1037e-05> 
[22-Jun-2022-Wed 00:32:28] INFO: <epoch: 22, iter: 5400, DT=64.00s, ET=6.40m> | ae: <lr: 6.250e-05; pixel_l1: 4.9398e-03, tv_2: 4.0206e-05> 
[22-Jun-2022-Wed 00:33:32] INFO: <epoch: 22, iter: 5500, DT=64.30s, ET=5.36m> | ae: <lr: 6.250e-05; pixel_l1: 4.8792e-03, tv_2: 4.0318e-05> 
[22-Jun-2022-Wed 00:34:37] INFO: <epoch: 23, iter: 5600, DT=65.20s, ET=4.35m> | ae: <lr: 6.250e-05; pixel_l1: 4.8778e-03, tv_2: 4.1260e-05> 
[22-Jun-2022-Wed 00:35:42] INFO: <epoch: 23, iter: 5700, DT=64.44s, ET=3.22m> | ae: <lr: 6.250e-05; pixel_l1: 4.8715e-03, tv_2: 4.0453e-05> 
[22-Jun-2022-Wed 00:36:48] INFO: <epoch: 24, iter: 5800, DT=65.86s, ET=2.20m> | ae: <lr: 6.250e-05; pixel_l1: 4.8140e-03, tv_2: 4.0917e-05> 
[22-Jun-2022-Wed 00:37:52] INFO: <epoch: 24, iter: 5900, DT=64.72s, ET=64.72s> | ae: <lr: 6.250e-05; pixel_l1: 4.7912e-03, tv_2: 4.1326e-05> 
[22-Jun-2022-Wed 00:38:57] INFO: <epoch: 24, iter: 6000, DT=64.25s, ET=0.00s> | ae: <lr: 6.250e-05; pixel_l1: 4.7384e-03, tv_2: 3.9426e-05> 
[22-Jun-2022-Wed 00:38:57] INFO: Checkpoint. Saving models...
[22-Jun-2022-Wed 00:38:57] INFO: Checkpoint. Saving models...
[22-Jun-2022-Wed 00:38:57] INFO: Training Classificator is ending...
```

## Last but not least

This project was heavily inspired by [traiNNer](https://github.com/victorca25/traiNNer) (developed by [victorca25](https://github.com/victorca25)). Try this one. They are also have a great Discord server "Enchance Everything". You will find the link in victorca25 rep. 
