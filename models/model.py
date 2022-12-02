import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import ResNet, ResNet_CIFAR


class ResNet(ResNet):
    def __init__(
        self,
        depth=50,
        num_classes=1000,
    ):
        super().__init__(
            depth=depth,
            num_classes=num_classes,
        )


class ResNet_CIFAR(ResNet_CIFAR):
    def __init__(
        self,
        depth=18,
        num_classes=100,
    ):
        super().__init__(
            depth=depth,
            num_classes=num_classes,
        )


