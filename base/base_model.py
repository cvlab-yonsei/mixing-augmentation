import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from torch.nn.parallel import DistributedDataParallel as DDP


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError
