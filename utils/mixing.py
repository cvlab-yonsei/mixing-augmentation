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


class Cutout_official():
    # Codes from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
        mix_prob (float)
        value (float)
    """
    def __init__(self, device='cpu', n_holes=1, length=8, mix_prob=1.0, value=0.):
        super().__init__()
        self.device = device
        self.n_holes = n_holes
        self.length = length

        self.mix_prob = mix_prob
        self.value = value

    def __str__(self):
        return "\n" + "-" * 10 + \
            f"\n** Official version of Cutout **\nAugmentation probability: {self.mix_prob}\n" + \
            "-" * 10

    def rand_bbox(self, size):
        W = size[2]
        H = size[3]

        # uniform
        cx = torch.randint(self.length // 2, (W - self.length // 2), (1,)).to(self.device)
        cy = torch.randint(self.length // 2, (H - self.length // 2), (1,)).to(self.device)

        bbx1 = torch.clip(cx - self.length // 2, 0, W)
        bby1 = torch.clip(cy - self.length // 2, 0, H)
        bbx2 = torch.clip(cx + self.length // 2, 0, W)
        bby2 = torch.clip(cy + self.length // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, image, target, model):
        """
        Args:
            img (Tensor): Tensor image of size (N, C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        mix_flag = False
        r = torch.rand(1).to(self.device)
        if r < self.mix_prob:
            mix_flag = True
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(image.shape)
            image[:, :, bbx1:bbx2, bby1:bby2] = self.value

            ratio = torch.ones(image.shape[0], device=self.device)
            
            return mix_flag, {"image": image, "target": (target, None), "ratio": ratio}
        else:
            return mix_flag, {}


class Cutout_lam():
    """Randomly mask out one or more patches from an image.
    Args:
        distribution (str)
        alpha (float)
        beta (float)
        mix_prob (float)
        value (float)
    """
    def __init__(self, device='cpu', distribution="beta", alpha1=1.0, alpha2=1.0, mix_prob=0.5, value=0):
        super().__init__()
        self.device = device
        self.distribution = distribution.lower()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.mix_prob = mix_prob
        self.value = value

        if self.distribution == "beta":
            self.sampler = torch.distributions.beta.Beta(torch.tensor([self.alpha1]), torch.tensor([self.alpha2]))
        elif self.distribution == "uniform":
            self.sampler = torch.distributions.uniform.Uniform(torch.tensor([self.alpha1]), torch.tensor([self.alpha2]))

    def __str__(self):
        return "\n" + "-" * 10 + \
            f"\n** Modified version of Cutout **\nlambda ~ {self.distribution.capitalize()}({self.alpha1}, {self.alpha2})\nmxing probability: {self.mix_prob}\n" + \
            "-" * 10

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        if self.distribution == "beta":
            cut_rat = torch.sqrt(1. - lam)
            cut_w = (W * cut_rat).int()
            cut_h = (H * cut_rat).int()
        elif self.distribution == "uniform":
            cut_w = (W * lam).int()
            cut_h = (H * lam).int()

        # uniform
        cx = torch.randint((cut_w // 2).item(), (W - cut_w // 2).item(), (1,)).to(cut_w.device)
        cy = torch.randint((cut_h // 2).item(), (H - cut_h // 2).item(), (1,)).to(cut_h.device)

        bbx1 = torch.clip(cx - cut_w // 2, 0, W)
        bby1 = torch.clip(cy - cut_h // 2, 0, H)
        bbx2 = torch.clip(cx + cut_w // 2, 0, W)
        bby2 = torch.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, image, target, model):
        mix_flag = False
        r = torch.rand(1).to(self.device)
        if (self.distribution != "none") and (r < self.mix_prob):
            mix_flag = True
            # generate mixed sample
            if self.distribution == "beta":
                lam = self.sampler.sample().to(self.device)
            elif self.distribution == "uniform":
                lam = self.sampler.sample().to(self.device)

            bbx1, bby1, bbx2, bby2 = self.rand_bbox(image.shape, lam)
            image[:, :, bbx1:bbx2, bby1:bby2] = self.value

            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
            ratio = torch.ones(image.shape[0], device=self.device) * lam

            return mix_flag, {"image": image, "target": (target, None), "ratio": ratio}
        else:
            return mix_flag, {}


class Cutout_m():
    """Randomly mask out one or more patches from an image.
    Args:
        distribution (str)
        alpha (float)
        beta (float)
        mix_prob (float)
        value (float)
    """
    def __init__(self, device='cpu', distribution="beta", alpha1=1.0, alpha2=1.0, mix_prob=0.5, value=0):
        super().__init__()
        self.device = device
        self.distribution = distribution.lower()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.mix_prob = mix_prob
        self.value = value
        if self.distribution == "beta":
            self.sampler = torch.distributions.beta.Beta(torch.tensor([self.alpha1]), torch.tensor([self.alpha2]))
        elif self.distribution == "uniform":
            self.sampler = torch.distributions.uniform.Uniform(torch.tensor([self.alpha1]), torch.tensor([self.alpha2]))

    def __str__(self):
        return "\n" + "-" * 10 + \
            f"\n** Modified version of Cutout **\nlambda ~ {self.distribution.capitalize()}({self.alpha1}, {self.alpha2})\nmxing probability: {self.mix_prob}\n" + \
            "-" * 10

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        if self.distribution == "beta":
            cut_rat = torch.sqrt(1. - lam)
            cut_w = (W * cut_rat).int()
            cut_h = (H * cut_rat).int()
        elif self.distribution == "uniform":
            cut_w = (W * lam).int()
            cut_h = (H * lam).int()

        # uniform
        cx = torch.randint((cut_w // 2).item(), (W - cut_w // 2).item(), (1,)).to(cut_w.device)
        cy = torch.randint((cut_h // 2).item(), (H - cut_h // 2).item(), (1,)).to(cut_h.device)

        bbx1 = torch.clip(cx - cut_w // 2, 0, W)
        bby1 = torch.clip(cy - cut_h // 2, 0, H)
        bbx2 = torch.clip(cx + cut_w // 2, 0, W)
        bby2 = torch.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, image, target, model):
        mix_flag = False
        r = torch.rand(1).to(self.device)
        if (self.distribution != "none") and (r < self.mix_prob):
            mix_flag = True
            # generate mixed sample
            if self.distribution == "beta":
                lam = self.sampler.sample().to(self.device)
            elif self.distribution == "uniform":
                lam = self.sampler.sample().to(self.device)

            bbx1, bby1, bbx2, bby2 = self.rand_bbox(image.shape, lam)
            image[:, :, bbx1:bbx2, bby1:bby2] = self.value

            ratio = torch.ones(image.shape[0], device=self.device)

            return mix_flag, {"image": image, "target": (target, None), "ratio": ratio}
        else:
            return mix_flag, {}


class CutMix():
    def __init__(self, device='cpu', distribution=None, alpha1=1.0, alpha2=1.0, mix_prob=0.5):
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
            f"\n** Official version of CutMix **\nlambda ~ {self.distribution.capitalize()}({self.alpha1}, {self.alpha2})\nmxing probability: {self.mix_prob}\n" + \
            "-" * 10

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = (W * cut_rat).int()
        cut_h = (H * cut_rat).int()

        # uniform
        cx = torch.randint(W, (1,)).to(cut_w.device)
        cy = torch.randint(H, (1,)).to(cut_h.device)

        bbx1 = torch.clip(cx - cut_w // 2, 0, W)
        bby1 = torch.clip(cy - cut_h // 2, 0, H)
        bbx2 = torch.clip(cx + cut_w // 2, 0, W)
        bby2 = torch.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, image, target, model):
        mix_flag = False
        r = torch.rand(1).to(self.device)
        if (self.distribution != "none") and (r < self.mix_prob):
            mix_flag = True
            # generate mixed sample
            if self.distribution == "beta":
                lam = self.sampler.sample().to(self.device)

            rand_index = torch.randperm(image.shape[0]).to(self.device)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(image.shape, lam)
            image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2, bby1:bby2]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
            ratio = torch.ones(image.shape[0], device=self.device) * lam

            return mix_flag, {"image": image, "target": (target, None), "ratio": ratio}
        else:
            return mix_flag, {}


class CutMix_m():
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
            f"\n** Modified version of CutMix **\nlambda ~ {self.distribution.capitalize()}({self.alpha1}, {self.alpha2})\nmxing probability: {self.mix_prob}\n" + \
            "-" * 10

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = (W * cut_rat).int()
        cut_h = (H * cut_rat).int()

        # uniform
        cx = torch.randint((cut_w // 2).item(), (W - cut_w // 2).item(), (1,)).to(cut_w.device)
        cy = torch.randint((cut_h // 2).item(), (H - cut_h // 2).item(), (1,)).to(cut_h.device)

        bbx1 = torch.clip(cx - cut_w // 2, 0, W)
        bby1 = torch.clip(cy - cut_h // 2, 0, H)
        bbx2 = torch.clip(cx + cut_w // 2, 0, W)
        bby2 = torch.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, image, target, model):
        mix_flag = False
        r = torch.rand(1).to(self.device)
        if (self.distribution != "none") and (r < self.mix_prob):
            mix_flag = True
            # generate mixed sample
            if self.distribution == "beta":
                lam = self.sampler.sample().to(self.device)

            rand_index = torch.randperm(image.shape[0]).to(self.device)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(image.shape, lam)
            image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2, bby1:bby2]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
            ratio = torch.ones(image.shape[0], device=self.device) * lam

            return mix_flag, {"image": image, "target": (target, target[rand_index]), "ratio": ratio}
        else:
            return mix_flag, {}