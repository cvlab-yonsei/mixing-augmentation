# import cv2
import gco  # For GraphCut algorithm in PuzzleMix
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from multiprocessing import Pool
from torch.autograd import Variable
from torch.nn.functional import interpolate


class Vanilla(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config

    def __str__(self):
        return "\n" + "-" * 10 + "** Vanilla **" + "-" * 10

    def forward(self, image, target, model):
        return False, {}

    def set_device(self, device):
        self.device = device


class AutoMix(Vanilla):
    def __init__(
        self, config, device='cpu', distribution='beta', alpha1=1.0, alpha2=1.0, layer_idex=3, mask_layer=2, mask_up_override=None,
        debug=False, pre_one_loss=0., lam_margin=-1, mask_loss=0., mask_adjust=0.
    ):
        super().__init__(config, device)
        self.distribution = str(distribution)
        self.layer_idex = int(layer_idex)
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)
        self.mask_layer = int(mask_layer)
        self.mask_up_override = mask_up_override \
            if isinstance(mask_up_override, (str, list)) else None
        if self.mask_up_override is not None:
            if isinstance(self.mask_up_override, str):
                self.mask_up_override = [self.mask_up_override]
            for m in self.mask_up_override:
                assert m in ['nearest', 'bilinear', 'bicubic',]
        self.debug = bool(debug)
        
        self.pre_one_loss = pre_one_loss
        self.lam_margin = float(lam_margin)
        self.mask_loss = mask_loss
        self.mask_adjust = float(mask_adjust)

    def __str__(self):
        return "\n" + "-" * 10 + "** AutoMix **" + "-" * 10

    def forward(self, image, target, backbone_k, mix_block):
        batch_size = image.shape[0]
        feature = backbone_k(image, extract_feature=True)

        if self.distribution == "beta":
            self.sampler = torch.distributions.beta.Beta(torch.tensor([self.alpha1]), torch.tensor([self.alpha2]))
            # np.random.beta(self.alpha1, self.alpha2, 2)
        elif self.distribution == "uniform":
            self.sampler = torch.distributions.uniform.Uniform(torch.tensor([self.alpha1]), torch.tensor([self.alpha2]))
            # np.random.random(self.alpha1, self.alpha2, 2)
        ratio = self.sampler.sample((2,)).to(image.device)
        # ratio = np.random.beta(self.alpha1, self.alpha2, 2)  # 0: mb, 1: bb

        index_mb = torch.randperm(batch_size).to(image.device)  # Used to train Encoder k
        index_bb = torch.randperm(batch_size).to(image.device)  # Used to train Encoder q

        indices = [index_mb, index_bb]
        results = self.pixel_mixup(image, target, ratio, indices, feature[self.layer_idex], mix_block)
        results['index_mb'] = index_mb
        results['index_bb'] = index_bb
        results['ratio'] = ratio
        
        return results

    def pixel_mixup(self, x, y, lam, index, feature, mix_block):
        """ pixel-wise input space mixup
        Args:
            x (Tensor): Input of a batch of images, (N, C, H, W).
            y (Tensor): A batch of gt_labels, (N, 1).
            lam (List): Input list of lambda (scalar).
            index (List): Input list of shuffle index (tensor) for mixup.
            feature (Tensor): The feature map of x, (N, C, H', W').
        Returns: dict includes following
            mixed_x_bb, mixed_x_mb: Mixup samples for bb (training the backbone)
                and mb (training the mixblock).
            mask_loss (Tensor): Output loss of mixup masks.
            pre_one_loss (Tensor): Output onehot cls loss of pre-mixblock.
        """
        results = dict()
        # lam info
        lam_mb = lam[0]  # lam is a scalar
        lam_bb = lam[1]

        # mask upsampling factor
        if x.shape[3] > 64:  # normal version of resnet
            scale_factor = 2**(2 + self.mask_layer)
        else:  # CIFAR version
            scale_factor = 2**self.mask_layer
        
        # get mixup mask
        mask_mb = mix_block(feature, lam_mb, index[0], scale_factor=scale_factor, debug=self.debug, unsampling_override=self.mask_up_override)
        mask_bb = mix_block(feature, lam_bb, index[1], scale_factor=scale_factor, debug=False, unsampling_override=None)
        if self.debug:
            results["debug_plot"] = mask_mb["debug_plot"]
        else:
            results["debug_plot"] = None

        # pre mixblock loss
        results["pre_one_loss"] = None
        if self.pre_one_loss > 0.:
            pred_one = mix_block.pre_head([mask_mb["x_lam"]])
            y_one = (y, y, 1)
            results["pre_one_loss"] = \
                mix_block.pre_head.loss(pred_one, y_one)["loss"] * self.pre_one_loss
            if torch.isnan(results["pre_one_loss"]):
                results["pre_one_loss"] = None
        
        mask_mb = mask_mb["mask"]
        mask_bb = mask_bb["mask"].clone().detach()

        # adjust mask_bb with lambd
        if self.mask_adjust > np.random.rand():  # [0,1)
            epsilon = 1e-8
            _mask = mask_bb[:, 0, :, :].squeeze()  # [N, H, W], _mask for lam
            _mask = _mask.clamp(min=epsilon, max=1 - epsilon)
            _mean = _mask.mean(dim=[1, 2]).squeeze()  # [N, 1, 1] -> [N]
            idx_larg = _mean[:] > lam[0] + epsilon  # index of mean > lam_bb
            idx_less = _mean[:] < lam[0] - epsilon  # index of mean < lam_bb
            # if mean > lam_bb
            mask_bb[idx_larg == True, 0, :, :] = _mask[idx_larg == True, :, :] * (lam[0] / _mean[idx_larg == True].view(-1, 1, 1))
            mask_bb[idx_larg == True, 1, :, :] = 1 - mask_bb[idx_larg == True, 0, :, :]
            # elif mean < lam_bb
            mask_bb[idx_less == True, 1, :, :] = (1 - _mask[idx_less == True, :, :]) * ((1 - lam[0]) / (1 - _mean[idx_less == True].view(-1, 1, 1)))
            mask_bb[idx_less == True, 0, :, :] = 1 - mask_bb[idx_less == True, 1, :, :]
        # lam_margin for backbone training
        if self.lam_margin >= lam_bb or self.lam_margin >= 1 - lam_bb:
            mask_bb[:, 0, :, :] = lam_bb
            mask_bb[:, 1, :, :] = 1 - lam_bb
        
        # loss of mixup mask
        results["mask_loss"] = None
        if self.mask_loss > 0.:
            if isinstance(mix_block, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                results["mask_loss"] = mix_block.module.mask_loss(mask_mb, lam_mb)["loss"]
            else:
                results["mask_loss"] = mix_block.mask_loss(mask_mb, lam_mb)["loss"]
            if results["mask_loss"] is not None:
                results["mask_loss"] *= self.mask_loss
        
        # mix, apply mask on x and x_
        # img_mix_mb = x * (1 - mask_mb) + x[index[0], :] * mask_mb
        assert mask_mb.shape[1] == 2
        assert mask_mb.shape[2:] == x.shape[2:], f"Invalid mask shape={mask_mb.shape}"
        results["img_mix_mb"] = x * mask_mb[:, 0, :, :].unsqueeze(1) + x[index[0], :] * mask_mb[:, 1, :, :].unsqueeze(1)
        
        # img_mix_bb = x * (1 - mask_bb) + x[index[1], :] * mask_bb
        results["img_mix_bb"] = x * mask_bb[:, 0, :, :].unsqueeze(1) + x[index[1], :] * mask_bb[:, 1, :, :].unsqueeze(1)
        
        return results


class Mixup(Vanilla):
    def __init__(self, config, device='cpu', distribution="beta", alpha1=1.0, alpha2=1.0, mix_prob=0.5):
        super().__init__(config, device)
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

    def forward(self, image, target, model):
        device = image.device
        mix_flag = False
        r = torch.rand(1).to(device)
        if (self.distribution != "none") and (r < self.mix_prob):
            mix_flag = True
            # generate mixed sample
            if self.distribution == "beta":
                lam = self.sampler.sample().to(device)

            rand_index = torch.randperm(image.shape[0]).to(device)
            image = lam * image + (1 - lam) * image[rand_index, :]

            ratio = torch.ones(image.shape[0], device=device) * lam

            return mix_flag, {"image": image, "target": (target, target[rand_index]), "ratio": ratio}
        else:
            return mix_flag, {}


class Cutout_official(Vanilla):
    # Codes from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
        mix_prob (float)
        value (float)
    """
    def __init__(self, config, device='cpu', n_holes=1, length=8, mix_prob=1.0, value=0.):
        super().__init__(config, device)
        self.n_holes = n_holes
        self.length = length

        self.mix_prob = mix_prob
        self.value = value

    def __str__(self):
        return "\n" + "-" * 10 + \
            f"\n** Official version of Cutout **\nPatch Length: {self.box_size}\nAugmentation probability: {self.mix_prob}\n" + \
            "-" * 10

    def rand_bbox(self, size, device):
        H, W = size[-2:]

        # uniform
        cy = torch.randint(self.length // 2, (H - self.length // 2).item(), (1,)).to(device)
        cx = torch.randint(self.length // 2, (W - self.length // 2).item(), (1,)).to(device)

        bby1 = torch.clip(cy - self.length // 2, 0, H)
        bbx1 = torch.clip(cx - self.length // 2, 0, W)
        bby2 = torch.clip(cy + self.length // 2, 0, H)
        bbx2 = torch.clip(cx + self.length // 2, 0, W)

        return bbx1, bby1, bbx2, bby2

    def forward(self, image, target, model):
        """
        Args:
            img (Tensor): Tensor image of size (N, C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        device = image.device
        mix_flag = False
        r = torch.rand(1).to(device)
        if r < self.mix_prob:
            mix_flag = True
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(image.shape, device)
            image[:, :, bby1:bby2, bbx1:bbx2] = self.value

            ratio = torch.ones(image.shape[0], device=device)
            
            return mix_flag, {"image": image, "target": (target, None), "ratio": ratio}
        else:
            return mix_flag, {}


class Cutout_m(Vanilla):
    """Randomly mask out one or more patches from an image.
    Args:
        distribution (str)
        alpha (float)
        beta (float)
        mix_prob (float)
        value (float)
    """
    def __init__(self, config, device='cpu', distribution="beta", alpha1=1.0, alpha2=1.0, mix_prob=0.5, value=0):
        super().__init__(config, device)
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
        H, W = size[-2:]
        if self.distribution == "beta":
            cut_rat = torch.sqrt(1. - lam)
            cut_w = (W * cut_rat).int()
            cut_h = (H * cut_rat).int()
        elif self.distribution == "uniform":
            cut_w = (W * lam).int()
            cut_h = (H * lam).int()

        # uniform
        cy = torch.randint((cut_h // 2).item(), (H - cut_h // 2).item(), (1,)).to(cut_h.device)
        cx = torch.randint((cut_w // 2).item(), (W - cut_w // 2).item(), (1,)).to(cut_w.device)

        bby1 = torch.clip(cy - cut_h // 2, 0, H)
        bbx1 = torch.clip(cx - cut_w // 2, 0, W)
        bby2 = torch.clip(cy + cut_h // 2, 0, H)
        bbx2 = torch.clip(cx + cut_w // 2, 0, W)

        return bbx1, bby1, bbx2, bby2

    def forward(self, image, target, model):
        device = image.device
        mix_flag = False
        r = torch.rand(1).to(device)
        if (self.distribution != "none") and (r < self.mix_prob):
            mix_flag = True
            # generate mixed sample
            if self.distribution == "beta":
                lam = self.sampler.sample().to(device)
            elif self.distribution == "uniform":
                lam = self.sampler.sample().to(device)

            bbx1, bby1, bbx2, bby2 = self.rand_bbox(image.shape, lam)
            image[:, :, bby1:bby2, bbx1:bbx2] = self.value

            ratio = torch.ones(image.shape[0], device=device)

            return mix_flag, {"image": image, "target": (target, None), "ratio": ratio}
        else:
            return mix_flag, {}


class CutMix(Vanilla):
    def __init__(self, config, device='cpu', distribution=None, alpha1=1.0, alpha2=1.0, mix_prob=0.5):
        super().__init__(config, device)
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
        H, W = size[-2:]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = (W * cut_rat).int()
        cut_h = (H * cut_rat).int()

        # uniform
        cy = torch.randint(H, (1,)).to(cut_h.device)
        cx = torch.randint(W, (1,)).to(cut_w.device)

        bby1 = torch.clip(cy - cut_h // 2, 0, H)
        bbx1 = torch.clip(cx - cut_w // 2, 0, W)
        bby2 = torch.clip(cy + cut_h // 2, 0, H)
        bbx2 = torch.clip(cx + cut_w // 2, 0, W)

        return bbx1, bby1, bbx2, bby2

    def forward(self, image, target, model):
        device = image.device
        mix_flag = False
        r = torch.rand(1).to(device)
        if (self.distribution != "none") and (r < self.mix_prob):
            mix_flag = True
            # generate mixed sample
            if self.distribution == "beta":
                lam = self.sampler.sample().to(device)

            rand_index = torch.randperm(image.shape[0]).to(device)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(image.shape, lam)
            image[:, :, bby1:bby2, bbx1:bbx2] = image[rand_index, :, bby1:bby2, bbx1:bbx2]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
            ratio = torch.ones(image.shape[0], device=device) * lam

            return mix_flag, {"image": image, "target": (target, None), "ratio": ratio}
        else:
            return mix_flag, {}


class CutMix_m(Vanilla):
    def __init__(self, config, device='cpu', distribution="beta", alpha1=1.0, alpha2=1.0, mix_prob=0.5):
        super().__init__(config, device)
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

    def rand_bbox(self, size, lam, device):
        H, W = size[-2:]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = (W * cut_rat).int()
        cut_h = (H * cut_rat).int()

        # uniform
        cy = torch.randint((cut_h // 2).item(), (H - cut_h // 2).item(), (1,)).to(device)
        cx = torch.randint((cut_w // 2).item(), (W - cut_w // 2).item(), (1,)).to(device)

        bby1 = torch.clip(cy - cut_h // 2, 0, H)
        bbx1 = torch.clip(cx - cut_w // 2, 0, W)
        bby2 = torch.clip(cy + cut_h // 2, 0, H)
        bbx2 = torch.clip(cx + cut_w // 2, 0, W)

        return bbx1, bby1, bbx2, bby2

    def forward(self, image, target, model):
        device = image.device
        mix_flag = False
        r = torch.rand(1).to(device)
        if (self.distribution != "none") and (r < self.mix_prob):
            mix_flag = True
            # generate mixed sample
            if self.distribution == "beta":
                lam = self.sampler.sample().to(device)

            rand_index = torch.randperm(image.shape[0]).to(device)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(image.shape, lam, device)
            image[:, :, bby1:bby2, bbx1:bbx2] = image[rand_index, :, bby1:bby2, bbx1:bbx2]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
            ratio = torch.ones(image.shape[0], device=device) * lam

            return mix_flag, {"image": image, "target": (target, target[rand_index]), "ratio": ratio}
        else:
            return mix_flag, {}


class ResizeMix(Vanilla):
    def __init__(self, config, device='cpu', distribution="uniform", alpha1=0.1, alpha2=0.8, mix_prob=0.5):
        super().__init__(config, device)
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

    def rand_bbox(self, size, tau, device):
        H, W = size[-2:]
        cut_w = (W * tau).int()
        cut_h = (H * tau).int()

        # uniform
        cy = torch.randint((cut_h // 2).item(), (H - cut_h // 2).item(), (1,)).to(device)
        cx = torch.randint((cut_w // 2).item(), (W - cut_w // 2).item(), (1,)).to(device)

        bby1 = torch.clip(cy - cut_h // 2, 0, H)
        bbx1 = torch.clip(cx - cut_w // 2, 0, W)
        bby2 = torch.clip(cy + cut_h // 2, 0, H)
        bbx2 = torch.clip(cx + cut_w // 2, 0, W)

        return bbx1, bby1, bbx2, bby2

    def forward(self, image, target, model):
        device = image.device
        mix_flag = False
        r = torch.rand(1).to(device)
        if (self.distribution != "none") and (r < self.mix_prob):
            mix_flag = True
            # generate mixed sample
            if self.distribution == "beta":
                tau = self.sampler.sample().to(device)
            elif self.distribution == "uniform":
                tau = self.sampler.sample().to(device)
            rand_index = torch.randperm(image.shape[0]).to(device)

            bbx1, bby1, bbx2, bby2 = self.rand_bbox(image.shape, tau, device)

            image_resize = interpolate(
                image.clone()[rand_index], (bby2 - bby1, bbx2 - bbx1), mode="nearest"
            )

            image[:, :, bby1:bby2, bbx1:bbx2] = image_resize

            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
            ratio = torch.ones(image.shape[0], device=device) * lam
            
            # return image, (target, target[rand_index]), lam
            return mix_flag, {"image": image, "target": (target, target[rand_index]), "ratio": ratio}
        else:
            return mix_flag, {}


class PuzzleMix(Vanilla):
    def __init__(
        self, config, device=None,
        distribution="beta", alpha1=1.0, alpha2=1.0, mix_prob=0.5,
        n_labels=3, beta=1.2, gamma=0.5, eta=0.2, neigh_size=4,
        transport=True, t_eps=0.8, t_size=-1, t_batch_size=16,
        adv_p=0, adv_eps=10., clean_lam=0., mp=8,
    ):
        super().__init__(config, device)
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
        self.t_batch_size = t_batch_size

        self.clean_lam = clean_lam
        self.adv_p = adv_p
        self.adv_eps = adv_eps

        if mp > 0:
            self.mp = Pool(mp)
        else:
            self.mp = None

        self.dataset = config['data_loader']['args']['dataset']
        if self.dataset == 'tiny-imagenet-200':
            self.mean = torch.tensor([0.5] * 3, dtype=torch.float32).reshape(1, 3, 1, 1)
            self.std = torch.tensor([0.5] * 3, dtype=torch.float32).reshape(1, 3, 1, 1)
            self.num_classes = 200
        elif self.dataset == 'cifar10':
            self.mean = torch.tensor([x / 255 for x in [125.3, 123.0, 113.9]], dtype=torch.float32).reshape(1, 3, 1, 1)
            self.std = torch.tensor([x / 255 for x in [63.0, 62.1, 66.7]], dtype=torch.float32).reshape(1, 3, 1, 1)
            self.num_classes = 10
        elif self.dataset == 'cifar100':
            self.mean = torch.tensor([x / 255 for x in [129.3, 124.1, 112.4]], dtype=torch.float32).reshape(1, 3, 1, 1)
            self.std = torch.tensor([x / 255 for x in [68.2, 65.4, 70.4]], dtype=torch.float32).reshape(1, 3, 1, 1)
            self.num_classes = 100
        elif self.dataset == 'imagenet':
            self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape(1, 3, 1, 1)

        self.cost_matrix_dict = {
            '2': self.cost_matrix(2).unsqueeze(0),
            '4': self.cost_matrix(4).unsqueeze(0),
            '8': self.cost_matrix(8).unsqueeze(0),
            '16': self.cost_matrix(16).unsqueeze(0)
        }

        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.deivce = 'cpu'

    def __str__(self):
        return "\n" + "-" * 10 + \
            f"\n** PuzzleMix (\u03B2, \u03B3, \u03B7, \u03B6) = ({self.beta}, {self.gamma}, {self.eta}, {self.t_eps}) **\nmxing probability: {self.mix_prob}\n" + \
            "-" * 10

    def forward(self, image, target, model):
        self.device = image.device
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)

        mix_flag = False
        r = torch.rand(1).to(self.device)
        if (self.distribution != "none") and (r < self.mix_prob):
            mix_flag = True
            unary, noise, adv_mask1, adv_mask2 = self.calculate_unary(image, target, model)
            image, rand_index, ratio = self.mixup_process(
                image,
                grad=unary,
                noise=noise,
                adv_mask1=adv_mask1,
                adv_mask2=adv_mask2,
                mp=self.mp
            )
            return mix_flag, {"image": image, "target": (target, target[rand_index]), "ratio": ratio}
        else:
            return mix_flag, {}
    
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
            image_orig = image * self.std.to(self.device) + self.mean.to(self.device)
            image_noise = image_orig + noise
            image_noise = torch.clamp(image_noise, 0, 1)
            noise = image_noise - image_orig
            image_noise = (image_noise - self.mean.to(self.device)) / self.std
            image_var = Variable(image_noise, requires_grad=True)
        else:
            noise = None
            image_var = Variable(image, requires_grad=True)
        target_var = Variable(target)

        # calculate saliency (unary)
        if self.clean_lam == 0:  # CIFAR 100
            model.eval()
            output = model(image_var)
            loss_clean = self.criterion(output, target_var)
            loss_clean.backward(retain_graph=False)
            model.zero_grad()  # using MCE only
            model.train()
        else:  # Tiny-imagenet & ImageNet
            model.train()
            output = model(image_var)
            if self.dataset == 'imagenet':
                loss_clean = self.clean_lam * self.criterion(output, target_var)  # https://github.com/snu-mllab/PuzzleMix/blob/e2dbf3a2371026411d5741d129f46bf3eb3d3465/imagenet/train.py#L385
            else:
                loss_clean = 2 * self.clean_lam * self.criterion(output, target_var) / self.num_classes  # https://github.com/snu-mllab/PuzzleMix/blob/e2dbf3a2371026411d5741d129f46bf3eb3d3465/main.py#L364
            loss_clean.backward(retain_graph=True)

        unary = torch.sqrt(torch.mean(image_var.grad**2, dim=1))

        # calculate adversarial noise
        if (adv_mask1 == 1 or adv_mask2 == 1):
            noise += (self.adv_eps + 2) / 255. * image_var.grad.sign()
            noise = torch.clamp(noise, -self.adv_eps / 255., self.adv_eps / 255.)
            adv_mix_coef = np.random.uniform(0, 1)
            noise = adv_mix_coef * noise
        
        return unary, noise, adv_mask1, adv_mask2

    def mixup_process(self, image, hidden=0, args=None, grad=None, noise=None, adv_mask1=0, adv_mask2=0, mp=None):
        block_num = 2**np.random.randint(1, 4)  # Following the AutoMix
        indices = torch.randperm(image.size(0)).to(self.device)

        lam = np.random.beta(self.alpha1, self.alpha2)
        image, ratio = self.mixup_graph(
            image,
            grad,
            indices=indices,
            block_num=block_num,
            alpha=lam,
            noise=noise,  # None for ImageNet
            adv_mask1=adv_mask1,  # 0 for ImageNet
            adv_mask2=adv_mask2,  # 0 for ImageNet
            mp=self.mp,
            beta=self.beta,
            gamma=self.gamma,
            eta=self.eta,
            neigh_size=self.neigh_size,
            n_labels=self.n_labels,
            transport=self.transport,
            t_eps=self.t_eps,
            t_size=self.t_size,
            t_batch_size=self.t_batch_size,
            mean=self.mean,
            std=self.std
        )
        return image, indices, ratio

    def mixup_graph(
        self, input1, grad1, indices, block_num=2, alpha=0.5,
        beta=0., gamma=0., eta=0.2, neigh_size=2, n_labels=2,
        mean=None, std=None, transport=False, t_eps=10.0,
        t_size=16, t_batch_size=16, noise=None, adv_mask1=0, adv_mask2=0,  # Not used when ImageNet
        mp=None,
    ):
        input2 = input1[indices].clone()

        batch_size, _, _, width = input1.shape
        block_size = width // block_num
        neigh_size = min(neigh_size, block_size)
        t_size = min(t_size, block_size)  # Not used when ImageNet

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

            plan1 = self.mask_transport(mask, unary1_torch, eps=t_eps, device=self.device)
            plan2 = self.mask_transport(1 - mask, unary2_torch, eps=t_eps, device=self.device)

            if self.t_batch_size is not None:
                # ImageNet
                t_batch_size = min(self.t_batch_size, 16)
                try:
                    for i in range((batch_size - 1) // t_batch_size + 1):
                        idx_from = i * t_batch_size
                        idx_to = min((i + 1) * t_batch_size, batch_size)
                        input1[idx_from:idx_to] = self.transport_image(input1[idx_from:idx_to], plan1[idx_from:idx_to], idx_to - idx_from, t_block_num, t_size)
                        input2[idx_from:idx_to] = self.transport_image(input2[idx_from:idx_to], plan2[idx_from:idx_to], idx_to - idx_from, t_block_num, t_size)
                except:
                    raise AssertionError(
                        "** GPU memory is lacking while transporting. Reduce the t_batch_size value in this function (mixup.transprort) **"
                    )
            else:
                input1 = self.transport_image(input1, plan1, batch_size, t_block_num, t_size)
                input2 = self.transport_image(input2, plan2, batch_size, t_block_num, t_size)

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

    def mask_transport(self, mask, grad_pool, eps=0.01, device='cpu'):
        '''optimal transport plan'''
        # batch_size = mask.shape[0]
        block_num = mask.shape[-1]

        n_iter = int(block_num)
        C = self.cost_matrix_dict[str(block_num)].to(device)

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

    def cost_matrix(self, width):
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

        return C


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
