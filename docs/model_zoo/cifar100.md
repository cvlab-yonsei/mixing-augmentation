# CIFAR 100
## Hardware specifications
* CPU: 2 x Intel Xeon Silver 4210R CPU @ 2.40GHz
* GPU: NVIDIA GeForce RTX 2080 Ti

## Training details
Model |  Optimizer | Batch size | Learning<br>rate | Weight<br>decay | Scheduler | Params(M) | #GPUs
:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---:
ResNet-18  | SGD | 100 | 0.1 | 1e-4 | Consine<br>annealing | 11.3 | 1  
ResNeXt-50<br>(32x4d) | SGD | 100 | 0.1 | 1e-4 | Consine<br>annealing | 23.4 | 1

## Quantitative results
### ResNet-18
#### Best Top-1 & Top-5 Acc. (%) / Running Time
Method | 200 epochs | 400 epochs | 800 epochs | 1200 epochs |
:--| :--: | :--: | :--: | :--: |
Vanilla                                                               | [76.56 / 92.95<br>(1h 59m)]() | [77.27 / 93.15<br>(3h 54m)]() | [77.97 / 93.50<br>(7h 49m)]() | [77.82 / 93.72<br>(11h 47m)]() |
Mixup (p=1.0)<br />[[ICLR '18](https://arxiv.org/abs/1710.09412)]     | [78.56 / 93.33<br>(1h 58m)]() | [80.02 / 93.42<br>(3h 56m)]() | [79.68 / 93.20<br>(7h 53m)]() | [80.16 / 92.97<br>(11h 55m)]() |
CutMix (p=0.5)<br />[[ICCV '19](https://arxiv.org/abs/1905.04899)]    | [79.46 / 94.36<br>(1h 57m)]() | [79.85 / 94.73<br>(3h 55m)]() | [80.46 / 94.72<br>(8h 00m)]() | [80.25 / 94.66<br>(11h 51m)]() |
ResizeMix (p=0.5)<br />[[arXiv '20](https://arxiv.org/abs/2012.11101)]| [79.32 / 94.24<br>(1h 57m)]() | [79.90 / 94.46<br>(3h 56m)]() | [79.95 / 94.39<br>(7h 46m)]() | [79.59 / 94.39<br>(11h 48m)]() |
PuzzleMix (p=0.5)<br />[[ICML '20](https://arxiv.org/abs/2009.06962)] | [79.11 / 94.26<br>(3h 17m)]() | [80.02 / 94.80<br>(6h 35m)]() | [81.12 / 95.36<br>(13h 12m)]() | [81.38 / 95.23<br>(20h 18m)]() |
PuzzleMix (p=1.0)<br />[[ICML '20](https://arxiv.org/abs/2009.06962)] | [79.63 / 94.46<br>(4h 39m)]() | [80.72 / 94.27<br>(9h 19m)]() | [81.25 / 94.72<br>(18h 42m)]() | [80.45 / 94.45<br>(1d 5h 11m)]() |

#### Median of Top-1 Acc. in the last 10 epochs (%) / Running Time
Method | 200 epochs | 400 epochs | 800 epochs | 1200 epochs |
:--| :--: | :--: | :--: | :--: |
Vanilla                                                               | 76.47<br>(1h 59m) | 77.14<br>(3h 54m) | 77.72<br>(7h 49m) | 77.65<br>(11h m) |
Mixup (p=1.0)<br />[[ICLR '18](https://arxiv.org/abs/1710.09412)]     | 78.16<br>(1h 58m) | 79.67<br>(3h 56m) | 79.22<br>(7h 53m) | 79.43<br>(11h 55m) |
CutMix (p=0.5)<br />[[ICCV '19](https://arxiv.org/abs/1905.04899)]    | 79.31<br>(1h 57m) | 79.72<br>(3h 55m) | 80.30<br>(8h 00m) | 80.11<br>(11h 51m) |
ResizeMix (p=0.5)<br />[[arXiv '20](https://arxiv.org/abs/2012.11101)]| 79.08<br>(1h 57m) | 79.56<br>(3h 56m) | 79.78<br>(7h 46m) | 79.44<br>(11h 48m) |
PuzzleMix (p=0.5)<br />[[ICML '20](https://arxiv.org/abs/2009.06962)] | 78.95<br>(3h 17m) | 79.90<br>(6h 35m) | 80.87<br>(13h 12m) | 81.12<br>(20h 18m) |
PuzzleMix (p=1.0)<br />[[ICML '20](https://arxiv.org/abs/2009.06962)] | 79.35<br>(4h 39m) | 80.52<br>(9h 19m) | 81.10<br>(18h 42m) | 80.27<br>(1d 5h 11m) |

### ResNet-50(32x4d)
#### Best Top-1 & Top-5 Acc. (%) / Running Time
Method | 200 epochs | 400 epochs | 800 epochs | 1200 epochs |
:--| :--: | :--: | :--: | :--: |
Vanilla                                                                | [79.33 / 94.52<br>(7h 11m)]() | [80.14 / 95.12<br>(14h 15m)]() | [80.88 / 94.90<br>(1d 4h 43m)]() | [81.24 / 95.33<br>(1d 18h 16m)]() |
Mixup (p=1.0)<br />[[ICLR '18](https://arxiv.org/abs/1710.09412)]      | [81.92 / 94.82<br>(7h 12m)]() | [82.21 / 94.19<br>(14h 20m)]() | [82.32 / 93.88<br>(1d 4h 45m)]() | [81.81 / 93.52<br>(1d 19h 01m)]() |
CutMix (p=0.5)<br />[[ICCV '19](https://arxiv.org/abs/1905.04899)]     | [82.22 / 95.13<br>(7h 11m)]() | [81.57 / 94.42<br>(14h 15m)]() | [81.60 / 94.72<br>(1d 4h 25m)]() | [80.64 / 94.22<br>(1d 17h 53m)]() |
ResizeMix (p=0.5)<br />[[arXiv '20](https://arxiv.org/abs/2012.11101)] | [81.62 / 95.41<br>(7h 10m)]() | [82.21 / 95.11<br>(14h 17m)]() | [80.79 / 94.22<br>(1d 4h 29m)]() | [80.21 / 94.19<br>(1d 18h 18m)]() |
PuzzleMix (p=0.5)<br />[[ICML '20](https://arxiv.org/abs/2009.06962)]  | [82.54 / 95.78<br>(11h 45m)]() | [83.21 / 96.27<br>(22h 42m)]() | [83.68 / 96.12<br>(1d 23h 00m)]() | [83.34 / 95.90<br>(2d 20h 12m)]() |
PuzzleMix (p=1.0)<br />[[ICML '20](https://arxiv.org/abs/2009.06962)]  | [82.80 / 95.58<br>(15h 03m)]() | [82.73 / 95.17<br>(1d 7h 31m)]() | [81.47 / 94.63<br>(2d 13h 04m)]() | [79.86 / 93.54<br>(3d 20h 27m)]() |

#### Median of Top-1 Acc. in the last 10 epochs (%) / Running Time
Method | 200 epochs | 400 epochs | 800 epochs | 1200 epochs |
:--| :--: | :--: | :--: | :--: |
Vanilla                                                                | 79.01<br>(7h 11m) | 79.89<br>(14h 15m) | 80.58<br>(1d 4h 43m) | 80.96<br>(1d 18h 16m) |
Mixup (p=1.0)<br />[[ICLR '18](https://arxiv.org/abs/1710.09412)]      | 81.27<br>(7h 12m) | 81.65<br>(14h 20m) | 81.42<br>(1d 4h 45m) | 81.22<br>(1d 19h 01m) |
CutMix (p=0.5)<br />[[ICCV '19](https://arxiv.org/abs/1905.04899)]     | 82.06<br>(7h 11m) | 81.10<br>(14h 15m) | 81.23<br>(1d 4h 25m) | 80.24<br>(1d 17h 53m) |
ResizeMix (p=0.5)<br />[[arXiv '20](https://arxiv.org/abs/2012.11101)] | 81.38<br>(7h 10m) | 82.02<br>(14h 17m) | 80.24<br>(1d 4h 29m) | 79.79<br>(1d 18h 18m) |
PuzzleMix (p=0.5)<br />[[ICML '20](https://arxiv.org/abs/2009.06962)]  | 82.37<br>(11h 45m) | 82.98<br>(22h 42m) | 83.43<br>(1d 23h 00m) | 83.11<br>(2d 20h 12m) |
PuzzleMix (p=1.0)<br />[[ICML '20](https://arxiv.org/abs/2009.06962)]  | 82.59<br>(15h 03m) | 82.18<br>(1d 7h 31m) | 80.94<br>(2d 13h 04m) | 79.06<br>(3d 20h 27m) |
