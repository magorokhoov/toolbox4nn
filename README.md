# Toolbox4nn
Toolbox4nn is PyTorch framework for research and developing. Priority is CV (classification, super-resolution, stylization, restoration, GAN, etc) but i think you can use it for NLP also.

*In near future I will make awesome readme and docs*
**README IS OLD! See src/option/train/ for actual option.yml**

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
    ├── modules -- losses, models, optimizers, schedulers, metrics, etc
    ├── options -- YAML confings for flexible, powerful and easy configuration
    ├── sh -- sh files start if you use linux like me
    ├── train.py -- main file
    └── utils -- useful stuff like loggers, nicer_timer, dict2str, etc
```
## YAML Config example
```yaml
name: class_5jul2022
task: classification
device: cuda # cpu cuda
use_amp: False
# gpu_ids: [0] will be released in next versions. Use device instead

dataset:
  name: folderfolders_classae
  path_dirs: /home/manavendra/homeNN/Datasets/mounted/Places205/images
  # /home/manavendra/homeNN/Datasets/mounted/FFHQ
  batch_size: 16
  shuffle: True
  num_workers: 4

networks:
  en:
    arch: classae_en
    in_nc: 3
    mid_nc: 32
    inner_nc: 256 # 64x8

    act_type: gelu
    norm_type: group
    norm_groups: 4
    
    optimizer:
      name: adamw
      weight_decay: 0.001
      beta1: 0.9
      beta2: 0.99
      lr: 1e-3

    scheduler:
      scheme: multistep
      #milestones: []
      milestones_rel: [0.2, 0.4, 0.6, 0.8] 
      gamma: 0.5
      
  class:
    arch: classae_class
    inner_nc: 256
    midclass_nc: 1024
    class_num: 205

    act_type: gelu
    norm_type: group
    norm_groups: 4
    dropout_rate: 0.5

    optimizer:
      name: adamw
      weight_decay: 0.001
      beta1: 0.9
      beta2: 0.99
      lr: 1e-3

    scheduler:
      scheme: multistep
      #milestones: []
      milestones_rel: [0.2, 0.4, 0.6, 0.8] 
      gamma: 0.5

lossers:
  l_class:
    losser_type: class
    loss:
      class:
        loss_name: CEL
        criterion: {criterion_type: CrossEntropyLoss}
        weight: 1.0

metrics: [acc]

weights: 
  #classae_en: 
  #  path: '../experiments/ae_faces_22jun2022/models/ae.pth'
  #  strict: False

experiments:
  root: '../experiments/'
  saving_freq: 2000
  display_freq: 100

train:
  n_iters: 10000
  
logger:
  print_freq: 500
  save_log_file: True
   
   
```

## Logger
```
[05-Jul-2022-Tue 20:47:45] INFO: Training initialization...
[05-Jul-2022-Tue 20:47:45] INFO: AMP disabled
[05-Jul-2022-Tue 20:47:45] INFO: Dataset [DatasetFFs - folderfolders] is created.
[05-Jul-2022-Tue 20:47:46] INFO: Dataset has 247266 images
[05-Jul-2022-Tue 20:47:46] INFO: Dataloader has 15454 mini-batches (batch_size=16)
[05-Jul-2022-Tue 20:47:46] INFO: Total epochs: 1, iters: 10000
[05-Jul-2022-Tue 20:47:46] INFO: Neural network parameters: 
[05-Jul-2022-Tue 20:47:46] INFO: en: 2,252,256
[05-Jul-2022-Tue 20:47:46] INFO: class: 3,975,513
[05-Jul-2022-Tue 20:47:46] INFO: 
[05-Jul-2022-Tue 20:47:46] INFO: Training started!
[05-Jul-2022-Tue 20:51:06] INFO: <epoch: 1, iter: 500, DT=3.34m, ET=63.45m> | en_lr=1.000e-03  class_lr=1.000e-03 | l_class: <CEL=5.3080e+00> metrics: <acc=7.6250e-03>  
[05-Jul-2022-Tue 20:54:13] INFO: <epoch: 1, iter: 1000, DT=3.12m, ET=56.08m> | en_lr=1.000e-03  class_lr=1.000e-03 | l_class: <CEL=5.2855e+00> metrics: <acc=6.5000e-03>  
[05-Jul-2022-Tue 20:57:10] INFO: <epoch: 1, iter: 1500, DT=2.96m, ET=50.30m> | en_lr=1.000e-03  class_lr=1.000e-03 | l_class: <CEL=5.2700e+00> metrics: <acc=5.7500e-03>  
[05-Jul-2022-Tue 21:00:30] INFO: <epoch: 1, iter: 2000, DT=3.32m, ET=53.12m> | en_lr=5.000e-04  class_lr=5.000e-04 | l_class: <CEL=5.1937e+00> metrics: <acc=8.8750e-03>  
[05-Jul-2022-Tue 21:00:30] INFO: Saving models...
[05-Jul-2022-Tue 21:03:27] INFO: <epoch: 1, iter: 2500, DT=2.95m, ET=44.30m> | en_lr=5.000e-04  class_lr=5.000e-04 | l_class: <CEL=5.0601e+00> metrics: <acc=1.7750e-02>  
[05-Jul-2022-Tue 21:07:19] INFO: <epoch: 1, iter: 3000, DT=3.88m, ET=54.27m> | en_lr=5.000e-04  class_lr=5.000e-04 | l_class: <CEL=4.9929e+00> metrics: <acc=1.9750e-02>  
[05-Jul-2022-Tue 21:10:18] INFO: <epoch: 1, iter: 3500, DT=2.98m, ET=38.70m> | en_lr=5.000e-04  class_lr=5.000e-04 | l_class: <CEL=4.9574e+00> metrics: <acc=2.3000e-02>  
[05-Jul-2022-Tue 21:13:16] INFO: <epoch: 1, iter: 4000, DT=2.97m, ET=35.66m> | en_lr=2.500e-04  class_lr=2.500e-04 | l_class: <CEL=4.9157e+00> metrics: <acc=2.7375e-02>  
[05-Jul-2022-Tue 21:13:16] INFO: Saving models...
[05-Jul-2022-Tue 21:16:14] INFO: <epoch: 1, iter: 4500, DT=2.96m, ET=32.61m> | en_lr=2.500e-04  class_lr=2.500e-04 | l_class: <CEL=4.8464e+00> metrics: <acc=3.2625e-02>  
[05-Jul-2022-Tue 21:19:11] INFO: <epoch: 1, iter: 5000, DT=2.95m, ET=29.49m> | en_lr=2.500e-04  class_lr=2.500e-04 | l_class: <CEL=4.8298e+00> metrics: <acc=2.8375e-02>  
[05-Jul-2022-Tue 21:22:08] INFO: <epoch: 1, iter: 5500, DT=2.95m, ET=26.51m> | en_lr=2.500e-04  class_lr=2.500e-04 | l_class: <CEL=4.8127e+00> metrics: <acc=3.3500e-02>  
[05-Jul-2022-Tue 21:25:05] INFO: <epoch: 1, iter: 6000, DT=2.95m, ET=23.57m> | en_lr=1.250e-04  class_lr=1.250e-04 | l_class: <CEL=4.7965e+00> metrics: <acc=3.7250e-02>  
[05-Jul-2022-Tue 21:25:05] INFO: Saving models...
[05-Jul-2022-Tue 21:28:02] INFO: <epoch: 1, iter: 6500, DT=2.95m, ET=20.64m> | en_lr=1.250e-04  class_lr=1.250e-04 | l_class: <CEL=4.7727e+00> metrics: <acc=3.6000e-02>  
[05-Jul-2022-Tue 21:30:58] INFO: <epoch: 1, iter: 7000, DT=2.95m, ET=17.69m> | en_lr=1.250e-04  class_lr=1.250e-04 | l_class: <CEL=4.7814e+00> metrics: <acc=3.3500e-02>  
[05-Jul-2022-Tue 21:33:55] INFO: <epoch: 1, iter: 7500, DT=2.95m, ET=14.73m> | en_lr=1.250e-04  class_lr=1.250e-04 | l_class: <CEL=4.7424e+00> metrics: <acc=3.8500e-02>  
[05-Jul-2022-Tue 21:36:51] INFO: <epoch: 1, iter: 8000, DT=2.93m, ET=11.70m> | en_lr=6.250e-05  class_lr=6.250e-05 | l_class: <CEL=4.7281e+00> metrics: <acc=4.0500e-02>  
[05-Jul-2022-Tue 21:36:51] INFO: Saving models...
[05-Jul-2022-Tue 21:39:46] INFO: <epoch: 1, iter: 8500, DT=2.92m, ET=8.77m> | en_lr=6.250e-05  class_lr=6.250e-05 | l_class: <CEL=4.7105e+00> metrics: <acc=4.0375e-02>  
[05-Jul-2022-Tue 21:42:42] INFO: <epoch: 1, iter: 9000, DT=2.92m, ET=5.85m> | en_lr=6.250e-05  class_lr=6.250e-05 | l_class: <CEL=4.7079e+00> metrics: <acc=4.0875e-02>  
[05-Jul-2022-Tue 21:45:37] INFO: <epoch: 1, iter: 9500, DT=2.92m, ET=2.92m> | en_lr=6.250e-05  class_lr=6.250e-05 | l_class: <CEL=4.6697e+00> metrics: <acc=4.3375e-02>  
[05-Jul-2022-Tue 21:48:33] INFO: <epoch: 1, iter: 10000, DT=2.93m, ET=0.00s> | en_lr=6.250e-05  class_lr=6.250e-05 | l_class: <CEL=4.6756e+00> metrics: <acc=4.4750e-02>  
[05-Jul-2022-Tue 21:48:33] INFO: Saving models...
[05-Jul-2022-Tue 21:48:33] INFO: Saving models...
[05-Jul-2022-Tue 21:48:33] INFO: Training is ending...

```

## Last but not least

This project was heavily inspired by [traiNNer](https://github.com/victorca25/traiNNer) (developed by [victorca25](https://github.com/victorca25)). Try this one. They are also have a great Discord server "Enchance Everything". You will find the link in victorca25 rep. 
