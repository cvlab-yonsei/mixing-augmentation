import math
import torch
from collections import Counter
from bisect import bisect_right
from typing import Dict, Any, List
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, num_epochs, iters_per_epoch, T_max=None, eta_min=0, last_epoch=-1, verbose=False):
        self.T_max = num_epochs if T_max is None else T_max
        self.eta_min = eta_min
        self.iters_per_epoch = iters_per_epoch
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / (self.T_max * self.iters_per_epoch))) / 2
                for base_lr in self.base_lrs]
