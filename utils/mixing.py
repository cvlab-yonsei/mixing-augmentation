# import cv2
import gco  # For GraphCut algorithm in PuzzleMix
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from multiprocessing import Pool
from torch.autograd import Variable
from torch.nn.functional import interpolate


class Vanilla():
    def __init__(self, config, device='cpu'):
        super().__init__()

    def __str__(self):
        return "\n" + "-" * 10 + "** Vanilla **" + "-" * 10

    def __call__(self, image, target, model):
        return False, {}


class Mixup():
    def __init__(self, config, device='cpu', distribution="beta", alpha1=1.0, alpha2=1.0, mix_prob=0.5):
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
    def __init__(self, config, device='cpu', n_holes=1, length=8, mix_prob=1.0, value=0.):
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
    def __init__(self, config, device='cpu', distribution="beta", alpha1=1.0, alpha2=1.0, mix_prob=0.5, value=0):
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
    def __init__(self, config, device='cpu', distribution="beta", alpha1=1.0, alpha2=1.0, mix_prob=0.5, value=0):
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
    def __init__(self, config, device='cpu', distribution=None, alpha1=1.0, alpha2=1.0, mix_prob=0.5):
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
    def __init__(self, config, device='cpu', distribution="beta", alpha1=1.0, alpha2=1.0, mix_prob=0.5):
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


class CutMix_m_forloop():
    def __init__(self, config, device='cpu', distribution="beta", alpha1=1.0, alpha2=1.0, mix_prob=0.5):
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
            ratio = torch.ones(image.shape[0], device=self.device)
            rand_index = torch.randperm(image.shape[0]).to(self.device)
            for batch_idx in range(image.shape[0]):
                lam = self.sampler.sample().to(self.device)
                bbx1, bby1, bbx2, bby2 = self.rand_bbox(image.shape, lam)
                image[batch_idx, :, bbx1:bbx2, bby1:bby2] = image[rand_index[batch_idx], :, bbx1:bbx2, bby1:bby2]

                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
                ratio[batch_idx] = lam

            return mix_flag, {"image": image, "target": (target, target[rand_index]), "ratio": ratio}
        else:
            return mix_flag, {}


class ResizeMix_m():
    def __init__(self, config, device='cpu', distribution="uniform", alpha1=0.1, alpha2=0.8, mix_prob=0.5):
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
            f"\n** ResizeMix **\ntau ~ {self.distribution.capitalize()}({self.alpha1}, {self.alpha2})\nmxing probability: {self.mix_prob}\n" + \
            "-" * 10

    def rand_bbox(self, size, tau):
        W = size[-2]
        H = size[-1]
        cut_w = (W * tau).int()
        cut_h = (H * tau).int()

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
                tau = self.sampler.sample().to(self.device)
            elif self.distribution == "uniform":
                tau = self.sampler.sample().to(self.device)
            rand_index = torch.randperm(image.shape[0]).to(self.device)

            bbx1, bby1, bbx2, bby2 = self.rand_bbox(image.shape, tau)

            if (bby2 - bby1 != 0) and (bbx2 - bbx1 != 0):
                image_resize = interpolate(
                    image.clone()[rand_index], (bby2 - bby1, bbx2 - bbx1), mode="nearest"
                )

                image[:, :, bbx1:bbx2, bby1:bby2] = image_resize

            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
            ratio = torch.ones(image.shape[0], device=self.device) * lam
            
            # return image, (target, target[rand_index]), lam
            return mix_flag, {"image": image, "target": (target, target[rand_index]), "ratio": ratio}
        else:
            return mix_flag, {}


def graphcut_multi(unary1, unary2, pw_x, pw_y, alpha, beta, eta, n_labels=2, eps=1e-8):
    '''alpha-beta swap algorithm'''
    block_num = unary1.shape[0]

    large_val = 1000 * block_num**2

    if n_labels == 2:
        prior = np.array([-np.log(alpha + eps), -np.log(1 - alpha + eps)])
    elif n_labels == 3:
        prior = np.array([
            -np.log(alpha**2 + eps), -np.log(2 * alpha * (1 - alpha) + eps),
            -np.log((1 - alpha)**2 + eps)
        ])
    elif n_labels == 4:
        prior = np.array([
            -np.log(alpha**3 + eps), -np.log(3 * alpha**2 * (1 - alpha) + eps),
            -np.log(3 * alpha * (1 - alpha)**2 + eps), -np.log((1 - alpha)**3 + eps)
        ])

    prior = eta * prior / block_num**2
    unary_cost = (large_val * np.stack([(1 - lam) * unary1 + lam * unary2 + prior[i] for i, lam in enumerate(np.linspace(0, 1, n_labels))], axis=-1)).astype(np.int32)
    pairwise_cost = np.zeros(shape=[n_labels, n_labels], dtype=np.float32)

    for i in range(n_labels):
        for j in range(n_labels):
            pairwise_cost[i, j] = (i - j)**2 / (n_labels - 1)**2

    pw_x = (large_val * (pw_x + beta)).astype(np.int32)
    pw_y = (large_val * (pw_y + beta)).astype(np.int32)
    labels = 1.0 - gco.cut_grid_graph(unary_cost, pairwise_cost, pw_x, pw_y, algorithm='swap') / (n_labels - 1)
    mask = labels.reshape(block_num, block_num)

    return mask