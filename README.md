# Mixing Augmentation

## Pre-requisites
This repository uses the following libraries:
* python (3.8.8)
* pytorch (1.8.1)
* torchvision (0.9.1)
* gco-wrapper (3.0.8) (https://github.com/Borda/pyGCO)

## Getting Started
### Datasets
#### Tiny-ImageNet
We use 100,000 training samples and 10,000 validation samples for Tiny-ImageNet. You can download the dataset by running the command [tinyimagenet.sh](dataset/tinyimagenet.sh) in your terminal. Once the download is complete, ensure that the data is organized in the following directory structure:
```bash
└── /dataset/tiny-imagenet-200
    ├── train/
    │   ├── n01443537
    │   │   ├── n01443537_0.JPEG
    │   │   ├── ...
    │   │   └── n01443537_99.JPEG
    │   ├── n01443537
    │   ├── ...
    │   └── n12267677
    ├── val/
    │   ├── n01443537
    │   │   ├── val_1230.JPEG
    │   │   ├── ...
    │   │   └── val_9949.JPEG
    │   ├── n01443537
    │   ├── ...
    │   ├── n12267677
    │   └── val_annotations.txt
    ├── test/
    └── wnids.txt
```

## Quantitative results 
### CIFAR 100
Method | ResNet18 | ResNext50
:--| :--: | :--:
Vanilla                                                             | 77.73<br>(7h 49m) | 80.58<br>(1d 4h 43m)
Mixup (p=1.0)<br>[[ICLR '18](https://arxiv.org/abs/1710.09412)]     | 79.22<br>(7h 53m) | 81.42<br>(1d 4h 45m)
CutMix (p=0.5)<br>[[ICCV '19](https://arxiv.org/abs/1905.04899)]    | 80.30<br>(8h 00m) | 81.23<br>(1d 4h 25m)
ResizeMix (p=0.5)<br>[[arXiv '20](https://arxiv.org/abs/2012.11101)]| 79.79<br>(7h 46m) | 80.24<br>(1d 4h 29m)
PuzzleMix (p=0.5)<br>[[ICML '20](https://arxiv.org/abs/2009.06962)] | 80.87<br>(13h 12m) | 83.43<br>(1d 23h 00m)
PuzzleMix (p=1.0)<br>[[ICML '20](https://arxiv.org/abs/2009.06962)] | 81.10<br>(18h 42m) | 80.94<br>(2d 13h 04m)

### Tiny-ImageNet
Method | ResNet18 | ResNext50
:--| :--: | :--:
Vanilla                                                             | 63.01<br>(20h 16m) | 65.91<br>(2d 5h 35m) 
Mixup (p=1.0)<br>[[ICLR '18](https://arxiv.org/abs/1710.09412)]     | 64.47<br>(20h 13m) | 67.48<br>(2d 4h 40m) 
CutMix (p=0.5)<br>[[ICCV '19](https://arxiv.org/abs/1905.04899)]    | 65.41<br>(19h 40m) | 67.83<br>(2d 8h 39m) 
ResizeMix (p=0.5)<br>[[arXiv '20](https://arxiv.org/abs/2012.11101)]| 65.34<br>(19h 41m) | 67.86<br>(2d 2h 02m) 
PuzzleMix (p=0.5)<br>[[ICML '20](https://arxiv.org/abs/2009.06962)] | 65.26<br>(1d 8h 29m) | 68.23<br>(3d 11h 48m) 
PuzzleMix (p=1.0)<br>[[ICML '20](https://arxiv.org/abs/2009.06962)] | 66.98<br>(1d 22h 14m) | 69.19<br>(4d 22h 23m) 


## Training
### CIFAR 100
* CutMix with Resnet-18
```Shell
#!/bin/bash
PORT='tcp://127.0.0.1:12345'
GPU=0
SAVEDIR='saved'
NAME="cutmix"
DATASET='cifar100'

python train.py -c configs/cifar100/resnet18/config_cutmix.json \
-d ${GPU} --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} --dataset ${DATASET}
```

### Tiny-ImageNet
* PuzzleMix with ResNeXt-50
```Shell
#!/bin/bash
PORT='tcp://127.0.0.1:12345'
GPU=0,1  # Using two GPUs
SAVEDIR='saved'
NAME="puzzlemix"
DATASET='tiny_imagenet'

python train.py -c configs/tiny_imagenet/resnext50-32x4d/config_puzzlemix.json \
-d ${GPU} --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} --dataset ${DATASET}
```

## Acknowledgements
* This template is borrowed from [pytorch-template](https://github.com/victoresque/pytorch-template).

## License
* This project is licensed under the GPL-3.0 license - see the [LICENSE](LICENSE) file for details
