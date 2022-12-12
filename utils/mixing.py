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


class PuzzleMix():
    def __init__(
        self, config, device=None,
        distribution="beta", alpha1=1.0, alpha2=1.0, mix_prob=0.5,
        n_labels=3, beta=1.2, gamma=0.5, eta=0.2, neigh_size=4,
        transport=True, t_eps=0.8, t_size=-1,
        adv_p=0, adv_eps=10., clean_lam=0., mp=8,
    ):
        super().__init__()
        self.device = device
        self.distribution = distribution.lower()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.mix_prob = mix_prob

        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.neigh_size = neigh_size
        self.n_labels = n_labels
        self.transport = transport
        self.t_eps = t_eps
        self.t_size = t_size

        self.clean_lam = clean_lam
        self.adv_p = adv_p
        self.adv_eps = adv_eps

        if mp > 0:
            self.mp = Pool(mp)
        else:
            self.mp = None

        dataset = config['data_loader']['args']['dataset']
        if dataset == 'tiny-imagenet-200':
            self.mean = torch.tensor([0.5] * 3, dtype=torch.float32).reshape(1, 3, 1, 1).to(self.device)
            self.std = torch.tensor([0.5] * 3, dtype=torch.float32).reshape(1, 3, 1, 1).to(self.device)
            self.labels_per_class = 500
            self.num_classes = 200
        elif dataset == 'cifar10':
            self.mean = torch.tensor([x / 255 for x in [125.3, 123.0, 113.9]], dtype=torch.float32).reshape(1, 3, 1, 1).to(self.device)
            self.std = torch.tensor([x / 255 for x in [63.0, 62.1, 66.7]], dtype=torch.float32).reshape(1, 3, 1, 1).to(self.device)
            self.labels_per_class = 5000
            self.num_classes = 10
        elif dataset == 'cifar100':
            self.mean = torch.tensor([x / 255 for x in [129.3, 124.1, 112.4]], dtype=torch.float32).reshape(1, 3, 1, 1).to(self.device)
            self.std = torch.tensor([x / 255 for x in [68.2, 65.4, 70.4]], dtype=torch.float32).reshape(1, 3, 1, 1).to(self.device)
            self.labels_per_class = 500
            self.num_classes = 100

        self.cost_matrix_dict = {
            '2': self.cost_matrix(2, self.device).unsqueeze(0),
            '4': self.cost_matrix(4, self.device).unsqueeze(0),
            '8': self.cost_matrix(8, self.device).unsqueeze(0),
            '16': self.cost_matrix(16, self.device).unsqueeze(0)
        }

        self.criterion_batch = nn.CrossEntropyLoss(reduction='none')

    def __str__(self):
        return "\n" + "-" * 10 + \
            f"\n** PuzzleMix (\u03B2, \u03B3, \u03B7, \u03B6) = ({self.beta}, {self.gamma}, {self.eta}, {self.t_eps}) **\nmxing probability: {self.mix_prob}\n" + \
            "-" * 10

    def __call__(self, image, target, model):
        mix_flag = False
        r = torch.rand(1).to(self.device)
        if (self.distribution != "none") and (r < self.mix_prob):
            mix_flag = True
            unary, noise, adv_mask1, adv_mask2 = self.calculate_unary(image, target, model)
            image_var = Variable(image)
            target_reweighted = self.to_one_hot(target, self.num_classes)
            image, rand_index, ratio = self.mixup_process(
                image_var,
                target_reweighted,
                grad=unary,
                noise=noise,
                adv_mask1=adv_mask1,
                adv_mask2=adv_mask2,
                mp=self.mp
            )
            return mix_flag, {"image": image, "target": (target, target[rand_index]), "ratio": ratio}
        else:
            return mix_flag, {}
            
    def mixup_process(
        self,
        image,
        target_reweighted,
        hidden=0,
        args=None,
        grad=None,
        noise=None,
        adv_mask1=0,
        adv_mask2=0,
        mp=None
    ):
        block_num = 2**np.random.randint(1, 5)
        indices = np.random.permutation(image.size(0))

        lam = np.random.beta(self.alpha1, self.alpha2)
        image, ratio = self.mixup_graph(
            image,
            grad,
            indices=indices,
            block_num=block_num,
            alpha=lam,
            noise=noise,
            adv_mask1=adv_mask1,
            adv_mask2=adv_mask2,
            mp=self.mp,
            beta=self.beta,
            gamma=self.gamma,
            eta=self.eta,
            neigh_size=self.neigh_size,
            n_labels=self.n_labels,
            transport=self.transport,
            t_eps=self.t_eps,
            t_size=self.t_size,
            mean=self.mean,
            std=self.std,
            device=self.device
        )
        return image, indices, ratio

    def calculate_unary(self, image, target, model):
        # whether to add adversarial noise or not
        if self.adv_p > 0:
            adv_mask1 = np.random.binomial(n=1, p=self.adv_p)
            adv_mask2 = np.random.binomial(n=1, p=self.adv_p)
        else:
            adv_mask1 = 0
            adv_mask2 = 0
        
        # random start
        if (adv_mask1 == 1 or adv_mask2 == 1):
            noise = torch.zeros_like(image).uniform_(-self.adv_eps / 255., self.adv_eps / 255.)
            image_orig = image * self.std + self.mean
            image_noise = image_orig + noise
            image_noise = torch.clamp(image_noise, 0, 1)
            noise = image_noise - image_orig
            image_noise = (image_noise - self.mean) / self.std
            image_var = Variable(image_noise, requires_grad=True)
        else:
            noise = None
            image_var = Variable(image, requires_grad=True)
        target_var = Variable(target)

        # calculate saliency (unary)
        if self.clean_lam == 0:
            model.eval()
            output = model(image_var)
            loss_batch = self.criterion_batch(output, target_var)
        else:
            model.train()
            output = model(image_var)
            loss_batch = 2 * self.clean_lam * self.criterion_batch(output, target_var) / self.num_classes

        loss_batch_mean = torch.mean(loss_batch, dim=0)
        loss_batch_mean.backward(retain_graph=True)

        unary = torch.sqrt(torch.mean(image_var.grad**2, dim=1))

        # calculate adversarial noise
        if (adv_mask1 == 1 or adv_mask2 == 1):
            noise += (self.adv_eps + 2) / 255. * image_var.grad.sign()
            noise = torch.clamp(noise, -self.adv_eps / 255., self.adv_eps / 255.)
            adv_mix_coef = np.random.uniform(0, 1)
            noise = adv_mix_coef * noise

        if self.clean_lam == 0:
            model.train()
        
        return unary, noise, adv_mask1, adv_mask2

    def mixup_graph(
        self, input1, grad1, indices, block_num=2, alpha=0.5,
        beta=0., gamma=0., eta=0.2, neigh_size=2, n_labels=2,
        mean=None, std=None, transport=False, t_eps=10.0, t_size=16,
        noise=None, adv_mask1=0, adv_mask2=0, device='cpu', mp=None
    ):
        input2 = input1[indices].clone()

        batch_size, _, _, width = input1.shape
        block_size = width // block_num
        neigh_size = min(neigh_size, block_size)
        t_size = min(t_size, block_size)

        # normalize
        beta = beta / block_num / 16

        # unary term
        grad1_pool = F.avg_pool2d(grad1, block_size)
        unary1_torch = grad1_pool / grad1_pool.reshape(batch_size, -1).sum(1).reshape(batch_size, 1, 1)
        unary2_torch = unary1_torch[indices]

        # calculate pairwise terms
        input1_pool = F.avg_pool2d(input1 * std + mean, neigh_size)
        input2_pool = input1_pool[indices]

        pw_x = torch.zeros([batch_size, 2, 2, block_num - 1, block_num], device=self.device)
        pw_y = torch.zeros([batch_size, 2, 2, block_num, block_num - 1], device=self.device)

        k = block_size // neigh_size

        pw_x[:, 0, 0], pw_y[:, 0, 0] = self.neigh_penalty(input2_pool, input2_pool, k)
        pw_x[:, 0, 1], pw_y[:, 0, 1] = self.neigh_penalty(input2_pool, input1_pool, k)
        pw_x[:, 1, 0], pw_y[:, 1, 0] = self.neigh_penalty(input1_pool, input2_pool, k)
        pw_x[:, 1, 1], pw_y[:, 1, 1] = self.neigh_penalty(input1_pool, input1_pool, k)

        pw_x = beta * gamma * pw_x
        pw_y = beta * gamma * pw_y

        # re-define unary and pairwise terms to draw graph
        unary1 = unary1_torch.clone()
        unary2 = unary2_torch.clone()

        unary2[:, :-1, :] += (pw_x[:, 1, 0] + pw_x[:, 1, 1]) / 2.
        unary1[:, :-1, :] += (pw_x[:, 0, 1] + pw_x[:, 0, 0]) / 2.
        unary2[:, 1:, :] += (pw_x[:, 0, 1] + pw_x[:, 1, 1]) / 2.
        unary1[:, 1:, :] += (pw_x[:, 1, 0] + pw_x[:, 0, 0]) / 2.

        unary2[:, :, :-1] += (pw_y[:, 1, 0] + pw_y[:, 1, 1]) / 2.
        unary1[:, :, :-1] += (pw_y[:, 0, 1] + pw_y[:, 0, 0]) / 2.
        unary2[:, :, 1:] += (pw_y[:, 0, 1] + pw_y[:, 1, 1]) / 2.
        unary1[:, :, 1:] += (pw_y[:, 1, 0] + pw_y[:, 0, 0]) / 2.

        pw_x = (pw_x[:, 1, 0] + pw_x[:, 0, 1] - pw_x[:, 1, 1] - pw_x[:, 0, 0]) / 2
        pw_y = (pw_y[:, 1, 0] + pw_y[:, 0, 1] - pw_y[:, 1, 1] - pw_y[:, 0, 0]) / 2

        unary1 = unary1.detach().cpu().numpy()
        unary2 = unary2.detach().cpu().numpy()
        pw_x = pw_x.detach().cpu().numpy()
        pw_y = pw_y.detach().cpu().numpy()

        # solve graphcut
        if mp is None:
            mask = []
            for i in range(batch_size):
                mask.append(self.graphcut_multi(unary2[i], unary1[i], pw_x[i], pw_y[i], alpha, beta, eta, n_labels))
        else:
            input_mp = []
            for i in range(batch_size):
                input_mp.append((unary2[i], unary1[i], pw_x[i], pw_y[i], alpha, beta, eta, n_labels))
            mask = mp.starmap(graphcut_multi, input_mp)

        # optimal mask
        mask = torch.tensor(mask, dtype=torch.float32, device=self.device)
        mask = mask.unsqueeze(1)

        # add adversarial noise
        if adv_mask1 == 1.:
            input1 = input1 * std + mean + noise
            input1 = torch.clamp(input1, 0, 1)
            input1 = (input1 - mean) / std

        if adv_mask2 == 1.:
            input2 = input2 * std + mean + noise[indices]
            input2 = torch.clamp(input2, 0, 1)
            input2 = (input2 - mean) / std

        # tranport
        if transport:
            if t_size == -1:
                t_block_num = block_num
                t_size = block_size
            elif t_size < block_size:
                # block_size % t_size should be 0
                t_block_num = width // t_size
                mask = F.interpolate(mask, size=t_block_num)
                grad1_pool = F.avg_pool2d(grad1, t_size)
                unary1_torch = grad1_pool / grad1_pool.reshape(batch_size, -1).sum(1).reshape(batch_size, 1, 1)
                unary2_torch = unary1_torch[indices]
            else:
                t_block_num = block_num

            # input1
            plan = self.mask_transport(mask, unary1_torch, eps=t_eps)
            input1 = self.transport_image(input1, plan, batch_size, t_block_num, t_size)

            # input2
            plan = self.mask_transport(1 - mask, unary2_torch, eps=t_eps)
            input2 = self.transport_image(input2, plan, batch_size, t_block_num, t_size)

        # final mask and mixed ratio
        mask = F.interpolate(mask, size=width)
        ratio = mask.reshape(batch_size, -1).mean(-1)

        return mask * input1 + (1 - mask) * input2, ratio

    def neigh_penalty(self, input1, input2, k):
        '''data local smoothness term'''
        pw_x = input1[:, :, :-1, :] - input2[:, :, 1:, :]
        pw_y = input1[:, :, :, :-1] - input2[:, :, :, 1:]

        pw_x = pw_x[:, :, k - 1::k, :]
        pw_y = pw_y[:, :, :, k - 1::k]

        pw_x = F.avg_pool2d(pw_x.abs().mean(1), kernel_size=(1, k))
        pw_y = F.avg_pool2d(pw_y.abs().mean(1), kernel_size=(k, 1))

        return pw_x, pw_y

    def mask_transport(self, mask, grad_pool, eps=0.01):
        '''optimal transport plan'''
        # batch_size = mask.shape[0]
        block_num = mask.shape[-1]

        n_iter = int(block_num)
        C = self.cost_matrix_dict[str(block_num)]

        z = (mask > 0).float()
        cost = eps * C - grad_pool.reshape(-1, block_num**2, 1) * z.reshape(-1, 1, block_num**2)

        # row and col
        for _ in range(n_iter):
            row_best = cost.min(-1)[1]
            plan = torch.zeros_like(cost).scatter_(-1, row_best.unsqueeze(-1), 1)

            # column resolve
            cost_fight = plan * cost
            col_best = cost_fight.min(-2)[1]
            plan_win = torch.zeros_like(cost).scatter_(-2, col_best.unsqueeze(-2), 1) * plan
            plan_lose = (1 - plan_win) * plan

            cost += plan_lose

        return plan_win

    def transport_image(self, img, plan, batch_size, block_num, block_size):
        '''apply transport plan to images'''
        input_patch = img.reshape([batch_size, 3, block_num, block_size, block_num * block_size]).transpose(-2, -1)
        input_patch = input_patch.reshape([batch_size, 3, block_num, block_num, block_size, block_size]).transpose(-2, -1)
        input_patch = input_patch.reshape([batch_size, 3, block_num**2, block_size, block_size]).permute(0, 1, 3, 4, 2).unsqueeze(-1)

        input_transport = plan.transpose(-2, -1).unsqueeze(1).unsqueeze(1).unsqueeze(1).matmul(input_patch).squeeze(-1).permute(0, 1, 4, 2, 3)
        input_transport = input_transport.reshape([batch_size, 3, block_num, block_num, block_size, block_size])
        input_transport = input_transport.transpose(-2, -1).reshape([batch_size, 3, block_num, block_num * block_size, block_size])
        input_transport = input_transport.transpose(-2, -1).reshape([batch_size, 3, block_num * block_size, block_num * block_size])

        return input_transport

    def cost_matrix(self, width, device='cpu'):
        '''transport cost'''
        C = np.zeros([width**2, width**2], dtype=np.float32)

        for m_i in range(width**2):
            i1 = m_i // width
            j1 = m_i % width
            for m_j in range(width**2):
                i2 = m_j // width
                j2 = m_j % width
                C[m_i, m_j] = abs(i1 - i2)**2 + abs(j1 - j2)**2

        C = C / (width - 1)**2
        C = torch.tensor(C)
        if device != 'cpu':
            C = C.to(device)

        return C

    def to_one_hot(self, inp, num_classes):
        '''one-hot label'''
        y_onehot = torch.zeros((inp.size(0), num_classes), dtype=torch.float32, device=inp.device)
        y_onehot.scatter_(1, inp.unsqueeze(1), 1)
        return y_onehot


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