# Tiny-ImageNet Benchmark
## Hardware specifications
* CPU: 2 x Intel Xeon Silver 4210R CPU @ 2.40GHz
* GPU: NVIDIA GeForce RTX 2080 Ti

## Training details
Model | Epochs|  Optimizer | Batch size | Learning<br>rate | Weight<br>decay | Scheduler | Params(M) | #GPUs
:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---:
ResNet-18  | 400 | SGD | 100 | 0.2 | 1e-4 | Consine<br>annealing | 11.3 | 1  
ResNeXt-50<br>(32x4d) | 400 | SGD | 100 | 0.2 | 1e-4 | Consine<br>annealing | 23.4 | 2 

## Quantitative results
### Median of Top-1 Acc. in the last 10 epochs (%) / Running Time
Method | ResNet18 | ResNext50(32x4d) |
:--| :--: | :--:
Vanilla                                                               | 63.01<br>(20h 16m) | 65.91<br>(2d 5h 35m) 
Mixup (p=1.0)<br>[[ICLR '18](https://arxiv.org/abs/1710.09412)]     | 64.47<br>(20h 13m) | 67.48<br>(2d 4h 40m) 
CutMix (p=0.5)<br>[[ICCV '19](https://arxiv.org/abs/1905.04899)]    | 65.41<br>(19h 40m) | 67.83<br>(2d 8h 39m) 
ResizeMix (p=0.5)<br>[[arXiv '20](https://arxiv.org/abs/2012.11101)]| 65.34<br>(19h 41m) | 67.86<br>(2d 2h 02m) 
PuzzleMix (p=1.0)<br>[[ICML '20](https://arxiv.org/abs/2009.06962)] | 66.98<br>(1d 22h 14m) | 69.19<br>(4d 22h 23m) 
<!-- PuzzleMix (p=0.5)<br>[[ICML '20](https://arxiv.org/abs/2009.06962)] | 65.26<br>(1d 8h 29m) | 68.23<br>(3d 11h 48m)  -->

### Best Top-1 & Top-5 Acc. (%)
Method | ResNet-18 | ResNext-50(32x4d) |
:--| :--: | :--:
Vanilla                                                             | 63.29 / 82.73<br>([weights]() / [config](https://github.com/cvlab-yonsei/mixing-augmentation/blob/main/configs/tiny_imagenet/resnet18/config_vanilla.json)) | 66.16 / 84.28<br>([weights]() / [config]()) 
Mixup (p=1.0)<br>[[ICLR '18](https://arxiv.org/abs/1710.09412)]     | 64.85 / 84.34<br>([weights]() / [config](https://github.com/cvlab-yonsei/mixing-augmentation/blob/main/configs/tiny_imagenet/resnet18/config_mixup.json)) | 67.93 / 85.11<br>([weights]() / [config]()) 
CutMix (p=0.5)<br>[[ICCV '19](https://arxiv.org/abs/1905.04899)]    | 65.84 / 84.57<br>([weights]() / [config](https://github.com/cvlab-yonsei/mixing-augmentation/blob/main/configs/tiny_imagenet/resnet18/config_cutmix.json)) | 68.07 / 85.34<br>([weights]() / [config]()) 
ResizeMix (p=0.5)<br>[[arXiv '20](https://arxiv.org/abs/2012.11101)]| 65.72 / 85.09<br>([weights]() / [config](https://github.com/cvlab-yonsei/mixing-augmentation/blob/main/configs/tiny_imagenet/resnet18/config_resizemix.json)) | 68.24 / 85.78<br>([weights]() / [config]()) 
PuzzleMix (p=1.0)<br>[[ICML '20](https://arxiv.org/abs/2009.06962)] | 67.22 / 85.85<br>([weights]() / [config](https://github.com/cvlab-yonsei/mixing-augmentation/blob/main/configs/tiny_imagenet/resnet18/config_puzzlemix.json)) | 69.56 / 86.84<br>([weights]() / [config]()) 
<!-- PuzzleMix (p=0.5)<br>[[ICML '20](https://arxiv.org/abs/2009.06962)] | 65.63 / 84.69<br>([weights]() / [config]()) | 68.50 / 86.08<br>([weights]() / [config]())  -->