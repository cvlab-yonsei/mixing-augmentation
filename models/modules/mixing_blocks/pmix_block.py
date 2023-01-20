import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.model as module_arch

from typing import Dict, Optional, Tuple, Union


class PixelMixBlock(nn.Module):
    """Pixel-wise MixBlock.

    Modified version of
        "AutoMix: Unveiling the Power of Mixup (https://arxiv.org/abs/2103.13027)"
        "Boosting Discriminative Visual Representation Learning with Scenario-Agnostic
            Mixup (https://arxiv.org/pdf/2111.15454.pdf)"

    *** Warning: FP16 training might result in `inf` or `nan`, please try a smaller
        batch size with FP32 when FP16 overflow occurs! ***
    """

    def __init__(
        self, in_channels, reduction=2, use_scale=True, unsampling_mode='bilinear',
        lam_concat=False, lam_concat_v=False, lam_mul=0., lam_residual=False, lam_mul_k=-1,
        value_neck_cfg=None, x_qk_concat=False, x_v_concat=False,
        att_norm_cfg=None, att_act_cfg=None,
        mask_loss_mode="L1", mask_loss_margin=0,
        frozen=False, init_cfg=None
    ):
        super().__init__()
        self.init_cfg = init_cfg
        # non-local args
        self.in_channels = int(in_channels)
        self.reduction = int(reduction)
        self.use_scale = bool(use_scale)
        self.inter_channels = max(in_channels // reduction, 1)
        self.unsampling_mode = [unsampling_mode] if isinstance(unsampling_mode, str) \
            else list(unsampling_mode)
        for m in self.unsampling_mode:
            assert m in ['nearest', 'bilinear', 'bicubic',]

        # mixblock args
        self.lam_concat = bool(lam_concat)
        self.lam_concat_v = bool(lam_concat_v)
        self.lam_mul = float(lam_mul) if float(lam_mul) > 0 else 0
        self.lam_mul_k = [lam_mul_k] if isinstance(lam_mul_k, (int, float)) else list(lam_mul_k)
        self.lam_residual = bool(lam_residual)
        assert att_norm_cfg is None or isinstance(att_norm_cfg, dict)
        assert att_act_cfg is None or isinstance(att_act_cfg, dict)
        assert value_neck_cfg is None or isinstance(value_neck_cfg, dict)
        self.value_neck_cfg = value_neck_cfg
        self.x_qk_concat = bool(x_qk_concat)
        self.x_v_concat = bool(x_v_concat)
        self.mask_loss_mode = str(mask_loss_mode)
        self.mask_loss_margin = max(mask_loss_margin, 0.)
        self.frozen = bool(frozen)
        assert 0 <= lam_mul and lam_mul <= 1
        for i in range(len(self.lam_mul_k)):
            self.lam_mul_k[i] = min(self.lam_mul_k[i], 10) if self.lam_mul_k[i] >= 0 else -1
        assert mask_loss_mode in ["L1", "L1+Variance", "L2+Variance", "Sparsity"]
        if self.lam_concat or self.lam_concat_v:
            assert self.lam_concat != self.lam_concat_v, \
                "lam_concat can be adopted on q,k,v or only on v"
        if self.lam_concat or self.lam_mul:
            assert self.lam_concat != self.lam_mul, \
                "both lam_concat and lam_mul change q,k,v in terms of lam"
        if self.lam_concat or self.x_qk_concat:
            assert self.lam_concat != self.x_qk_concat, \
                "x_lam=x_lam_=cat(x,x_) if x_qk_concat=True, it's no use to concat lam"
        # FP16 training: exit after 10 times overflow
        self.overflow = 0

        # concat all as k,q,v
        self.qk_in_channels = int(in_channels + 1) \
            if self.lam_concat else int(in_channels)
        self.v_in_channels = int(in_channels + 1) \
            if self.lam_concat or self.lam_concat_v else int(in_channels)
        if self.x_qk_concat:
            self.qk_in_channels = int(2 * self.in_channels)
        if self.x_v_concat:
            self.v_in_channels = int(2 * self.in_channels)

        # MixBlock, conv value
        if value_neck_cfg is None:
            self.value = nn.Conv2d(
                in_channels=self.v_in_channels,
                out_channels=1,
                kernel_size=1,
                stride=1)
        else:
            value_neck_cfg["args"]["in_channels"] = self.v_in_channels
            self.value = getattr(module_arch, value_neck_cfg['type'])(**dict(value_neck_cfg['args']))
        # MixBlock, conv q,k
        self.key = None
        if self.x_qk_concat:  # sym conv q and k
            # conv key
            self.key = self._make_layer(
                in_channels=self.qk_in_channels,
                out_channels=self.inter_channels,
                kernel_size=1, stride=1, padding=0,
                groups=1, bias='auto',
                norm_cfg=att_norm_cfg,
                act_cfg=att_act_cfg,
            )
        # conv query
        self.query = self._make_layer(
            in_channels=self.qk_in_channels,
            out_channels=self.inter_channels,
            kernel_size=1, stride=1, padding=0,
            groups=1, bias='auto',
            norm_cfg=att_norm_cfg,
            act_cfg=att_act_cfg,
        )

        self.init_weights()
        if self.frozen:
            self._freeze()
    
    def _make_layer(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False, norm_cfg=None, act_cfg=None):
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=self.qk_in_channels,
                out_channels=self.inter_channels,
                kernel_size=1, stride=1, padding=0,
                groups=1, bias=(norm_cfg is None),
            )
        )
        if norm_cfg is not None:
            norm_cfg['args']['num_features'] = self.inter_channels
            layers.append(
                getattr(torch.nn, norm_cfg['type'])(**dict(norm_cfg['args']))
            )
        if act_cfg is not None:
            layers.append(
                getattr(torch.nn, act_cfg['type'])(**dict(act_cfg['args']))
            )
        return nn.Sequential(*layers)

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        if self.init_cfg is not None:
            super(PixelMixBlock, self).init_weights()
            return
        assert init_linear in ['normal', 'kaiming'], \
            "Undefined init_linear: {}".format(init_linear)
        # init mixblock
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if init_linear == 'normal':
                    nn.init.normal_(m.weight, std=std)
                    nn.init.constant_(m.bias, bias)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _freeze(self):
        if self.frozen:
            # mixblock
            for param in self.query.parameters():
                param.requires_grad = False
            if self.key is not None:
                for param in self.key.parameters():
                    param.requires_grad = False
            for param in self.value.parameters():
                param.requires_grad = False

    def embedded_gaussian(self, q_x, k_x):
        """Caculate learnable non-local similarity.

        Notice: force fp32 before and after matmul in attention, since
            fp16 will cause inf or nan without any pre-normalization.
            NonLocal2d pairwise_weight: [N, HxW, HxW].
        """
        pairwise_weight = torch.matmul(
            q_x.type(torch.float32), k_x.type(torch.float32)
        ).type(torch.float32)
        if torch.any(torch.isnan(pairwise_weight)):
            raise ValueError
        if torch.any(torch.isinf(pairwise_weight)):
            pairwise_weight = pairwise_weight.type(torch.float32).clamp(min=-1e25, max=1e25)
            self.overflow += 1
            if self.overflow > 10:
                raise ValueError("Precision overflow in MixBlock, try fp32 training.")
        if self.use_scale:
            # q_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= q_x.shape[-1] ** 0.5
        # force fp32 in exp
        pairwise_weight = pairwise_weight.type(torch.float32).softmax(dim=-1)
        return pairwise_weight

    def rescale_lam_mult(self, lam, k=1):
        """ adjust lam against y=x in terms of k """
        assert k >= 0
        k += 1
        lam = float(lam)
        return 1 / (k - 2 / 3) * (4 / 3 * math.pow(lam, 3) - 2 * lam**2 + k * lam)

    def forward(self, x, lam, index, scale_factor, debug=False, unsampling_override=None):
        """
        Args:
            x (tensor): Input feature map [N, C, H, W].
            lam (int): Mixup ratio lambda.
            index (tensor): Random shuffle index in current mini-batch.
            scale_factor (int): Unsampling factor (assert scale_factor % 2 == 0).
            debug (bool): Whether to use debug mode.
            unsampling_override (optional): Override upsampling mode for MixBlock.
        """
        results = dict()
        # pre-step 0: input 2d feature map x, [N, C, H, W]
        if isinstance(x, list) and index is None:
            assert len(x) == 2  # only for SSL mixup
            x = torch.cat(x)
        n, _, h, w = x.size()

        if index is None:  # only for SSL mixup, [2N, C, H, W]
            n = n // 2
            x_lam  = x[:n, ...]
            x_lam_ = x[n:, ...]
        else:  # supervised cls
            x_lam  = x
            x_lam_ = x[index, :]  # shuffle within a gpu
        results = dict(x_lam=x_lam, x_lam_=x_lam_)

        # pre-step 1: lambda encoding
        if self.lam_mul > 0:  # multiply lam to x_lam
            assert self.lam_concat is False
            # rescale lam
            _lam_mul_k = random.choices(self.lam_mul_k, k=1)[0]
            if _lam_mul_k >= 0:
                lam_rescale = self.rescale_lam_mult(lam, _lam_mul_k)
            else:
                lam_rescale = lam
            # using residual
            if self.lam_residual:
                x_lam = x_lam * (1 + lam_rescale * self.lam_mul)
                x_lam_ = x_lam_ * (1 + (1 - lam_rescale) * self.lam_mul)
            else:
                x_lam = x_lam * lam_rescale
                x_lam_ = x_lam_ * (1 - lam_rescale)
        if self.lam_concat:  # concat lam as a new channel
            # assert self.lam_mul > 0 and self.x_qk_concat == False
            lam_block = torch.zeros(n, 1, h, w).to(x_lam)
            lam_block[:] = lam
            x_lam  = torch.cat([x_lam, lam_block], dim=1)
            x_lam_ = torch.cat([x_lam_, 1 - lam_block], dim=1)

        # **** step 1: conpute 1x1 conv value, v: [N, HxW, 1] ****
        v_ = x_lam_
        if self.x_v_concat:
            v_ = torch.cat([x_lam, x_lam_], dim=1)
        if self.lam_concat_v:
            lam_block = torch.zeros(n, 1, h, w).to(x_lam)
            lam_block[:] = lam
            v_ = torch.cat([x_lam_, 1 - lam_block], dim=1)
        # compute v_
        if self.value_neck_cfg is None:
            v_ = self.value(v_).view(n, 1, -1)  # [N, 1, HxW]
        else:
            v_ = self.value([v_])[0].view(n, 1, -1)  # [N, 1, HxW]
        v_ = v_.permute(0, 2, 1)  # v_ for 1-lam: [N, HxW, 1]

        # debug mode
        if debug:
            debug_plot = dict(value=v_.view(n, h, -1).clone().detach())

        # **** step 2: compute 1x1 conv q & k, q_x: [N, HxW, C], k_x: [N, C, HxW] ****
        if self.x_qk_concat:
            x_lam = torch.cat([x_lam, x_lam_], dim=1)
            x_lam_ = x_lam
        # query
        q_x = self.query(x_lam).view(  # q for lam: [N, HxW, C/r]
            n, self.inter_channels, -1).permute(0, 2, 1)
        # key
        if self.key is not None:
            k_x = self.key(x_lam_).view(n, self.inter_channels, -1)  # [N, C/r, HxW]
        else:
            k_x = self.query(x_lam_).view(n, self.inter_channels, -1)  # [N, C/r, HxW]

        # **** step 3: 2d pairwise_weight: [N, HxW, HxW] ****
        pairwise_weight = self.embedded_gaussian(q_x, k_x)  # x_lam [N, HxW, C/r] x [N, C/r, HxW] x_lam_

        # debug mode
        if debug:
            debug_plot["pairwise_weight"] = pairwise_weight.clone().detach()
            results["debug_plot"] = debug_plot

        # choose upsampling mode
        if unsampling_override is not None:
            if isinstance(unsampling_override, str):
                up_mode = unsampling_override
            elif isinstance(unsampling_override, list):
                up_mode = random.choices(unsampling_override, k=1)[0]
            else:
                up_mode = "nearest"
        else:
            up_mode = random.choices(self.unsampling_mode, k=1)[0]

        # **** step 4: generate mixup mask and upsampling ****
        # P x v_lam_ = mask_lam_, force fp32 in matmul (causing NAN in fp16)
        mask_lam_ = torch.matmul(
            pairwise_weight.type(torch.float32), v_.type(torch.float32)
        ).view(n, 1, h, w)  # mask for 1-lam
        if torch.any(torch.isnan(mask_lam_)):
            mask_lam_ = torch.matmul(
                pairwise_weight.type(torch.float64), v_.type(torch.float64)
            ).view(n, 1, h, w)
            mask_lam_ = torch.where(torch.isnan(mask_lam_),
                                    torch.full_like(mask_lam_, 1e-4), mask_lam_)
        mask_lam_ = F.interpolate(mask_lam_, scale_factor=scale_factor, mode=up_mode)
        # mask for 1-lam in [0, 1], force fp32 in exp (causing NAN in fp16)
        mask_lam_ = torch.sigmoid(mask_lam_.type(torch.float32))

        mask = torch.cat([1 - mask_lam_, mask_lam_], dim=1)

        results["mask"] = mask
        return results

    def mask_loss(self, mask, lam):
        """ loss for mixup masks """
        losses = dict()
        assert mask.dim() == 4
        n, k, h, w = mask.size()  # mixup mask [N, 2, H, W]
        if k > 1:  # the second mask has no grad!
            mask = mask[:, 1, :, :].unsqueeze(1)
        m_mean = mask.sum() / (n * h * w)  # mask mean in [0, 1]

        if self.mask_loss_mode == "L1":  # [0, 1-m]
            losses['loss'] = torch.clamp(
                torch.abs(1 - m_mean - lam) - self.mask_loss_margin, min=0.).mean()
        elif self.mask_loss_mode == "Sparsity":  # [0, 0.25-m]
            losses['loss'] = torch.clamp(
                torch.abs(mask * (mask - 1)).sum() / (n * h * w) - self.mask_loss_margin, min=0.)
        elif self.mask_loss_mode == "L1+Variance":  # [0, 1-m] + [0, 1]
            losses['loss'] = torch.clamp(
                torch.abs(1 - m_mean - lam) - self.mask_loss_margin, min=0.).mean() - \
                2 * torch.clamp((torch.sum((mask - m_mean)**2) / (n * h * w)), min=0.)
        elif self.mask_loss_mode == "L2+Variance":  # [0, 1-m^2] + [0, 1]
            losses['loss'] = torch.clamp(
                (1 - m_mean - lam) ** 2 - self.mask_loss_margin ** 2, min=0.).mean() - \
                2 * torch.clamp((torch.sum((mask - m_mean)**2) / (n * h * w)), min=0.)
        else:
            raise NotImplementedError
        if torch.isnan(losses['loss']):
            losses['loss'] = None
            self.overflow += 1
            if self.overflow > 10:
                raise ValueError("Precision overflow in MixBlock, try fp32 training.")
        return losses
