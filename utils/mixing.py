# import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from multiprocessing import Pool
from torch.autograd import Variable
from torch.nn.functional import interpolate


class Vanilla():
    def __init__(self, device='cpu'):
        super().__init__()

    def __str__(self):
        return "\n" + "-" * 10 + "** Vanilla **" + "-" * 10

    def __call__(self, image, target, model):
        return False, {}


class Mixup():
    def __init__(self, device='cpu', distribution="beta", alpha1=1.0, alpha2=1.0, mix_prob=0.5):
        super().__init__()
        self.device = device
        self.distribution = distribution.lower()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.mix_prob = mix_prob
        if self.distribution == "beta":
            self.sampler = torch.distributions.beta.Beta(torch.tensor([self.alpha1]), torch.tensor([self.alpha2]))
        elif self.distribution == "uniform":
            self.sampler = torch.distributions.uniform.Uniform(torch.tensor([self.alpha1]), torch.tensor([self.alpha2]))

    def __str__(self):
        return "\n" + "-" * 10 + \
            f"\n** MixUp **\nlambda ~ {self.distribution.capitalize()}({self.alpha1}, {self.alpha2})\nmxing probability: {self.mix_prob}\n" + \
            "-" * 10

    def __call__(self, image, target, model):
        mix_flag = False
        r = torch.rand(1).to(self.device)
        if (self.distribution != "none") and (r < self.mix_prob):
            mix_flag = True
            # generate mixed sample
            if self.distribution == "beta":
                lam = self.sampler.sample().to(self.device)

            rand_index = torch.randperm(image.shape[0]).to(self.device)
            image = lam * image + (1 - lam) * image[rand_index, :]

            ratio = torch.ones(image.shape[0], device=self.device) * lam

            return mix_flag, {"image": image, "target": (target, target[rand_index]), "ratio": ratio}
        else:
            return mix_flag, {}

