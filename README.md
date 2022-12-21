# Mixing Augmentation

## Pre-requisites
This repository uses the following libraries:
* python (3.8.8)
* pytorch (1.8.1)
* torchvision (0.9.1)
* gco-wrapper (3.0.8) (https://github.com/Borda/pyGCO)

## Getting Started

## Download Checkpoints
### CIFAR 100

<details open>
<summary><b>ResNet 18</b></summary>

#### Training details
    Batch size     : 100
    Optimizier     : SGD
    Learning rate  : 0.1
    Weight decay   : 1e-4
    Scheduler      : Cosine annealing
    Number of GPUs : 1

#### Quantitative results (Best Top-1 & Top-5 Acc. (%) / Running Time)
Method | 200 epochs | 400 epochs | 800 epochs | 1200 epochs |
:--| :--: | :--: | :--: | :--: |
Vanilla                                                               | [76.56 / 92.95<br>(1h 59m)]() | [77.27 / 93.15<br>(3h 54m)]() | [77.97 / 93.50<br>(7h 49m)]() | [77.82 / 93.72<br>(11h 47m)]() |
Mixup (p=1.0)<br />[[ICLR '18](https://arxiv.org/abs/1710.09412)]     | [78.56 / 93.33<br>(1h 58m)]() | [80.02 / 93.42<br>(3h 56m)]() | [79.68 / 93.20<br>(7h 53m)]() | [80.16 / 92.97<br>(11h 55m)]() |
CutMix (p=0.5)<br />[[ICCV '19](https://arxiv.org/abs/1905.04899)]    | [79.46 / 94.36<br>(1h 57m)]() | [79.85 / 94.73<br>(3h 55m)]() | [80.46 / 94.72<br>(8h 00m)]() | [80.25 / 94.66<br>(11h 51m)]() |
ResizeMix (p=0.5)<br />[[arXiv '20](https://arxiv.org/abs/2012.11101)]| [79.32 / 94.24<br>(1h 57m)]() | [79.90 / 94.46<br>(3h 56m)]() | [79.95 / 94.39<br>(7h 46m)]() | [79.59 / 94.39<br>(11h 48m)]() |
PuzzleMix (p=0.5)<br />[[ICML '20](https://arxiv.org/abs/2009.06962)] | [79.11 / 94.26<br>(3h 17m)]() | [80.02 / 94.80<br>(6h 35m)]() | [81.12 / 95.36<br>(13h 12m)]() | [81.38 / 95.23<br>(20h 18m)]() |
PuzzleMix (p=1.0)<br />[[ICML '20](https://arxiv.org/abs/2009.06962)] | [79.63 / 94.46<br>(4h 39m)]() | [80.72 / 94.27<br>(9h 19m)]() | [81.25 / 94.72<br>(18h 42m)]() | [80.45 / 94.45<br>(1d 5h 11m)]() |


#### Quantitative results (Median of Top-1 Acc. in the last 10 epochs (%) / Running Time)
Method | 200 epochs | 400 epochs | 800 epochs | 1200 epochs |
:--| :--: | :--: | :--: | :--: |
Vanilla                                                               | 76.47<br>(1h 59m) | 77.14<br>(3h 54m) | 77.72<br>(7h 49m) | 77.65<br>(11h m) |
Mixup (p=1.0)<br />[[ICLR '18](https://arxiv.org/abs/1710.09412)]     | 78.16<br>(1h 58m) | 79.67<br>(3h 56m) | 79.22<br>(7h 53m) | 79.43<br>(11h 55m) |
CutMix (p=0.5)<br />[[ICCV '19](https://arxiv.org/abs/1905.04899)]    | 79.31<br>(1h 57m) | 79.72<br>(3h 55m) | 80.30<br>(8h 00m) | 80.11<br>(11h 51m) |
ResizeMix (p=0.5)<br />[[arXiv '20](https://arxiv.org/abs/2012.11101)]| 79.08<br>(1h 57m) | 79.56<br>(3h 56m) | 79.78<br>(7h 46m) | 79.44<br>(11h 48m) |
PuzzleMix (p=0.5)<br />[[ICML '20](https://arxiv.org/abs/2009.06962)] | 78.95<br>(3h 17m) | 79.90<br>(6h 35m) | 80.87<br>(13h 12m) | 81.12<br>(20h 18m) |
PuzzleMix (p=1.0)<br />[[ICML '20](https://arxiv.org/abs/2009.06962)] | 79.35<br>(4h 39m) | 80.52<br>(9h 19m) | 81.10<br>(18h 42m) | 80.27<br>(1d 5h 11m) |
</details>


<details open>
<summary><b>ResNeXt 50 (32x4d)</b></summary>

#### Training details
    Batch size     : 100
    Optimizier     : SGD
    Learning rate  : 0.1
    Weight decay   : 1e-4
    Scheduler      : Cosine annealing
    Number of GPUs : 1

#### Quantitative results (Best Top-1 & Top-5 Acc. (%) / Running Time)
Method | 200 epochs | 400 epochs | 800 epochs | 1200 epochs |
:--| :--: | :--: | :--: | :--: |
Vanilla                                                                | [79.33 / 94.52<br>(7h 11m)]() | [ / <br>()]() | [ / <br>()]() | [ / <br>()]() |
Mixup (p=1.0)<br />[[ICLR '18](https://arxiv.org/abs/1710.09412)]      | [81.92 / 94.82<br>(7h 12m)]() | [ / <br>()]() | [ / <br>()]() | [ / <br>()]() |
CutMix (p=0.5)<br />[[ICCV '19](https://arxiv.org/abs/1905.04899)]     | [82.22 / 95.13<br>(7h 11m)]() | [ / <br>()]() | [ / <br>()]() | [ / <br>()]() |
ResizeMix (p=0.5)<br />[[arXiv '20](https://arxiv.org/abs/2012.11101)] | [81.62 / 95.41<br>(7h 10m)]() | [ / <br>()]() | [ / <br>()]() | [ / <br>()]() |
PuzzleMix (p=0.5)<br />[[ICML '20](https://arxiv.org/abs/2009.06962)]  | [82.54 / 95.78<br>(11h 45m)]() | [ / <br>()]() | [ / <br>()]() | [ / <br>()]() |
PuzzleMix (p=1.0)<br />[[ICML '20](https://arxiv.org/abs/2009.06962)]  | [82.80 / 95.58<br>(15h 03m)]() | [ / <br>()]() | [ / <br>()]() | [ / <br>()]() |


#### Quantitative results (Median of Top-1 Acc. in the last 10 epochs (%) / Running Time)
Method | 200 epochs | 400 epochs | 800 epochs | 1200 epochs |
:--| :--: | :--: | :--: | :--: |
Vanilla                                                                | 79.01<br>(7h 11m) | <br>() | <br>() | <br>() |
Mixup (p=1.0)<br />[[ICLR '18](https://arxiv.org/abs/1710.09412)]      | 81.27<br>(7h 12m) | <br>() | <br>() | <br>() |
CutMix (p=0.5)<br />[[ICCV '19](https://arxiv.org/abs/1905.04899)]     | 82.06<br>(7h 11m) | <br>() | <br>() | <br>() |
ResizeMix (p=0.5)<br />[[arXiv '20](https://arxiv.org/abs/2012.11101)] | 81.38<br>(7h 10m) | <br>() | <br>() | <br>() |
PuzzleMix (p=0.5)<br />[[ICML '20](https://arxiv.org/abs/2009.06962)]  | 82.37<br>(11h 45m) | <br>() | <br>() | <br>() |
PuzzleMix (p=1.0)<br />[[ICML '20](https://arxiv.org/abs/2009.06962)]  | 82.59<br>(15h 03m) | <br>() | <br>() | <br>() |
</details>


## Training
### CIFAR 100
* Vanilla
```Shell
python train.py -c configs/cifar100/resnet18/config_widresnet_vanilla.json \
-d 0--name "vanilla"
```
* PuzzleMix
```Shell
python train.py -c configs/cifar100/resnet18/config_widresnet_puzzlemix.json \
-d 0 --name "puzzlemix"
```

## Acknowledgements
* This template is borrowed from [pytorch-template](https://github.com/victoresque/pytorch-template).
