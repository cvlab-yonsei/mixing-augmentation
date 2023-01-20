import numpy as np
import torch
import torch.nn as nn
import utils.mixing as module_mixing

from models.modules import modules as module_backbone
from models.modules.mixing_blocks import pmix_block


class Mixup(nn.Module):
    def __init__(self, backbone, mixing_augmentation=None, num_classes=None, config=None):
        super().__init__()
        # Backbone
        backbone['args'].update({"num_classes": num_classes})
        self.backbone = getattr(module_backbone, backbone['type'])(**dict(backbone['args']))

        # Mixing augmentation
        mixing_augmentation['args'].update({"config": config})
        self.mixing_augmentation = getattr(module_mixing, mixing_augmentation['type'])(**dict(mixing_augmentation['args']))
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, img, trg=None, inference=False):
        if inference is True or trg is None:
            return self.backbone(img)

        output = {}
        mix_flag, mix_dict = self.mixing_augmentation(img, trg, self.backbone)
        if mix_flag is False:
            # Vanilla
            output['logit'] = self.backbone(img)
            output['total_loss'] = self.criterion(output['logit'], trg).mean()
        else:
            # Mixup
            output['logit'] = self.backbone(mix_dict['image'])
            if mix_dict['target'][1] is None:
                output['total_loss'] = (self.criterion(output['logit'], mix_dict['target'][0]) * mix_dict['ratio']).mean()
            else:
                output['target'] = self.criterion(output['logit'], mix_dict['target'][0]) * mix_dict['ratio']
                output['source'] = self.criterion(output['logit'], mix_dict['target'][1]) * (1. - mix_dict['ratio'])
                output['total_loss'] = (output['target'] + output['source']).mean()

        return output

    def mixing_info(self):
        return str(self.mixing_augmentation)

    def set_device(self, device):
        self.device = device
        self.mixing_augmentation.set_device(device)


class AutoMix(nn.Module):
    def __init__(
        self, backbone, backbone_k=None, mixing_augmentation=None, mix_block=None, head_mix=None, head_one=None, head_mix_k=None, head_one_k=None,
        head_weights=dict(decent_weight=[], accent_weight=[], head_mix_q=1, head_one_q=1, head_mix_k=1, head_one_k=1),
        momentum=0.999, mask_loss=0., mask_adjust=0., lam_margin=-1, debug=False,
        pre_one_loss=0., pre_mix_loss=0., switch_off=0., head_ensemble=False, save=False, save_name='MixedSamples',
        mix_shuffle_no_repeat=False, pretrained=None, pretrained_k=None, init_cfg=None, num_classes=1000, config=None
    ):
        super().__init__()
        # basic params
        self.momentum = float(momentum)
        self.base_momentum = float(momentum)
        self.mask_loss = float(mask_loss) if float(mask_loss) > 0 else 0
        self.mask_adjust = float(mask_adjust)
        self.pre_one_loss = float(pre_one_loss) if float(pre_one_loss) > 0 else 0
        self.pre_mix_loss = float(pre_mix_loss) if float(pre_mix_loss) > 0 else 0
        self.lam_margin = float(lam_margin) if float(lam_margin) > 0 else 0
        self.switch_off = float(switch_off) if float(switch_off) > 0 else 0
        self.head_ensemble = bool(head_ensemble)
        
        self.save = bool(save)
        self.save_name = str(save_name)
        self.debug = bool(debug)
        self.mix_shuffle_no_repeat = bool(mix_shuffle_no_repeat)
        assert 0 <= self.momentum and self.lam_margin < 1 and self.mask_adjust <= 1
        self.num_classes = num_classes

        # network
        assert isinstance(mix_block, dict) and isinstance(backbone, dict)
        assert backbone_k is None or isinstance(backbone_k, dict)
        assert head_mix is None or isinstance(head_mix, dict)
        assert head_one is None or isinstance(head_one, dict)
        assert head_mix_k is None or isinstance(head_mix_k, dict)
        assert head_one_k is None or isinstance(head_one_k, dict)
        head_mix_k = head_mix if head_mix_k is None else head_mix_k
        head_one_k = head_one if head_one_k is None else head_one_k

        # backbone
        self.backbone_q = getattr(module_backbone, backbone['type'])(**dict(backbone['args']))
        if backbone_k is not None:
            self.backbone_k = getattr(module_backbone, backbone_k['type'])(**dict(backbone_k['args']))
            assert self.momentum >= 1. and pretrained_k is not None
        else:
            self.backbone_k = getattr(module_backbone, backbone['type'])(**dict(backbone['args']))
        self.backbone = self.backbone_k  # for feature extract

        # mixblock
        self.mix_block = getattr(pmix_block, mix_block['type'])(**dict(mix_block['args']))
        
        '''
        # mixup cls head
        assert "head_mix_q" in head_weights.keys() and "head_mix_k" in head_weights.keys()
        
        self.head_mix_q = builder.build_head(head_mix)
        self.head_mix_k = builder.build_head(head_mix_k)

        # onehot cls head
        if "head_one_q" in head_weights.keys():
            self.head_one_q = builder.build_head(head_one)
        else:
            self.head_one_q = None
        if "head_one_k" in head_weights.keys() and "head_one_q" in head_weights.keys():
            self.head_one_k = builder.build_head(head_one_k)
        else:
            self.head_one_k = None

        # for feature extract
        self.head = self.head_one_k if self.head_one_k is not None else self.head_one_q
        # onehot and mixup heads for training
        '''
        self.weight_mix_q = head_weights.get("head_mix_q", 1.)
        self.weight_mix_k = head_weights.get("head_mix_k", 1.)
        self.weight_one_q = head_weights.get("head_one_q", 1.)
        assert self.weight_mix_q > 0 and (self.weight_mix_k > 0 or backbone_k is not None)
        self.head_weights = head_weights
        self.head_weights['decent_weight'] = head_weights.get("decent_weight", list())
        self.head_weights['accent_weight'] = head_weights.get("accent_weight", list())
        self.head_weights['mask_loss'] = self.mask_loss
        self.head_weights['pre_one_loss'] = self.pre_one_loss
        self.head_weights['pre_mix_loss'] = self.pre_mix_loss
        self.cos_annealing = 1.  # decent from 1 to 0 as cosine

        self._init_weights(pretrained=pretrained, pretrained_k=pretrained_k)

        # Mixing augmentation
        mixing_augmentation['args'].update(dict(config=config, debug=self.debug, pre_one_loss=self.pre_one_loss, lam_margin=self.lam_margin, mask_loss=self.mask_loss, mask_adjust=self.mask_adjust))
        self.mixing_augmentation = getattr(module_mixing, mixing_augmentation['type'])(**mixing_augmentation['args'])

        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def _init_weights(self, pretrained=None, pretrained_k=None):
        """Initialize the weights of model.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
            pretrained_k (str, optional): Path to pre-trained weights to initialize the
                backbone_k and mixblock. Default: None.
        """
        # init mixblock
        if self.mix_block is not None:
            self.mix_block.init_weights(init_linear='normal')

        # init pretrained backbone_k and mixblock
        if pretrained_k is not None:
            # load full ckpt to backbone and fc
            '''
            load_checkpoint(self, pretrained_k, strict=False, logger=logger)
            '''
            self.backbone_k._load_pretrained_model(pretrained_k, strict=False)
            '''
            # head_mix_k and head_one_k should share the same initalization
            if self.head_mix_k is not None and self.head_one_k is not None:
                for param_one_k, param_mix_k in zip(self.head_one_k.parameters(), self.head_mix_k.parameters()):
                    param_mix_k.data.copy_(param_one_k.data)
                    param_mix_k.requires_grad = False  # stop grad k
            '''

        # init backbone, based on params in q
        if pretrained is not None:
            self.backbone_q.init_weights(pretrained=pretrained)

        # copy backbone param from q to k
        if pretrained_k is None and self.momentum < 1:
            for param_q, param_k in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False  # stop grad k
        '''
        # init head
        if self.head_mix_q is not None:
            self.head_mix_q.init_weights()
        if self.head_one_q is not None:
            self.head_one_q.init_weights()
        
        # copy head one param from q to k
        if (self.head_one_q is not None and self.head_one_k is not None) and (pretrained_k is None and self.momentum < 1):
            for param_one_q, param_one_k in zip(self.head_one_q.parameters(), self.head_one_k.parameters()):
                param_one_k.data.copy_(param_one_q.data)
                param_one_k.requires_grad = False  # stop grad k
        # copy head mix param from q to k
        if (self.head_mix_q is not None and self.head_mix_k is not None) and (pretrained_k is None and self.momentum < 1):
            for param_mix_q, param_mix_k in zip(self.head_mix_q.parameters(), self.head_mix_k.parameters()):
                param_mix_k.data.copy_(param_mix_q.data)
                param_mix_k.requires_grad = False  # stop grad k
        '''

    def forward(self, img, gt_label=None, inference=False):
        self.momentum_update()

        if inference is True or gt_label is None:
            return self.backbone_q(img)

        mix_dict = self.mixing_augmentation(img, gt_label, self.backbone_k, self.mix_block)
        # k (mb): the mix block training
        loss_mix_k = self.forward_k(mix_dict["img_mix_mb"], gt_label, mix_dict['index_mb'], mix_dict['ratio'][0])
        # q (bb): the encoder training
        loss_one_q, loss_mix_q = self.forward_q(img, mix_dict["img_mix_bb"].detach(), gt_label, mix_dict['index_bb'], mix_dict['ratio'][1])
        
        # loss summary
        losses = {
            "total_loss": loss_mix_q["loss"] * self.weight_mix_q,
            # 'acc_mix_q': loss_mix_q["acc"],
        }

        # onehot loss
        if loss_one_q is not None and self.weight_one_q > 0:
            losses["total_loss"] += loss_one_q["loss"] * self.weight_one_q
            # losses['acc_one_q'] = loss_one_q["acc"]
        # mixblock loss
        if loss_mix_k["loss"] is not None and self.weight_mix_k > 0:
            losses["total_loss"] += loss_mix_k["loss"] * self.weight_mix_k
            # losses['acc_mix_k'] = loss_mix_k["acc"]
        # else:
        #     losses['acc_mix_k'] = loss_mix_q["acc"]

        if mix_dict["mask_loss"] is not None and self.mask_loss > 0:
            losses["total_loss"] += mix_dict["mask_loss"]

        if mix_dict["pre_one_loss"] is not None and self.pre_one_loss > 0:
            losses["total_loss"] += mix_dict["pre_one_loss"]
        if loss_mix_k["pre_mix_loss"] is not None and self.pre_mix_loss > 0:
            losses["total_loss"] += loss_mix_k["pre_mix_loss"]

        return losses

    def forward_k(self, mixed_x, y, index, lam):
        """ forward k with the mixup sample """
        loss_mix_k = dict(loss=None, pre_mix_loss=None)
        # switch off mixblock training
        if self.switch_off > 0:  # default: 0, 0.8 for deit and pvt
            if 0 < self.cos_annealing <= 1:
                if np.random.rand() > self.switch_off * self.cos_annealing:
                    return loss_mix_k
        
        # training mixblock from k
        if self.weight_mix_k > 0:  # self.weight_mix_k = head_weights.get("head_mix_k", 1.), default: 1
            # mixed_x forward
            '''
            out_mix_k = self.backbone_k(mixed_x)
            pred_mix_k = self.head_mix_k([out_mix_k[-1]])
            # force fp32 in mixup loss (causing NAN in fp16 training with a large batch size)
            pred_mix_k[0] = pred_mix_k[0].type(torch.float32)
            '''
            pred_mix_k = self.backbone_k(mixed_x).type(torch.float32)
            '''
            # k mixup loss
            y_mix_k = (y, y[index], lam)
            loss_mix_k = self.head_mix_k.loss(pred_mix_k, y_mix_k)
            '''
            # k mixup loss
            loss_mix_k['loss'] = self.criterion(pred_mix_k, y) * lam + self.criterion(pred_mix_k, y[index]) * (1. - lam)
            loss_mix_k['loss'] = loss_mix_k['loss'].mean()
            if torch.isnan(loss_mix_k["loss"]):
                '''
                print_log("Warming NAN in loss_mix_k. Please use FP32!", logger='root')
                '''
                loss_mix_k["loss"] = None
        '''
        # mixup loss, short cut of pre-mixblock
        if self.pre_mix_loss > 0:  # self.pre_mix_loss = float(pre_mix_loss) if float(pre_mix_loss) > 0 else 0, default: 0
            out_mb = out_mix_k[0]
            # pre FFN
            if self.mix_block.pre_attn is not None:
                out_mb = self.mix_block.pre_attn(out_mb)  # non-local
            if self.mix_block.pre_conv is not None:
                out_mb = self.mix_block.pre_conv([out_mb])  # neck
            # pre mixblock mixup loss
            pred_mix_mb = self.mix_block.pre_head(out_mb)
            # force fp32 in mixup loss (causing NAN in fp16 training with a large batch size)
            pred_mix_mb[0] = pred_mix_mb[0].type(torch.float32)
            loss_mix_k["pre_mix_loss"] = \
                self.mix_block.pre_head.loss(pred_mix_mb, y_mix_k)["loss"] * self.pre_mix_loss
            if torch.isnan(loss_mix_k["pre_mix_loss"]):
                print_log("Warming NAN in pre_mix_loss.", logger='root')
                loss_mix_k["pre_mix_loss"] = None
        else:
            loss_mix_k["pre_mix_loss"] = None
        '''
        return loss_mix_k

    def forward_q(self, x, mixed_x, y, index, lam):
        """
        Args:
            x (Tensor): Input of a batch of images, (N, C, H, W).
            mixed_x (Tensor): Mixup images of x, (N, C, H, W).
            y (Tensor): Groundtruth onehot labels, coresponding to x.
            index (List): Input list of shuffle index (tensor) for mixup.
            lam (List): Input list of lambda (scalar).
        Returns:
            dict[str, Tensor]: loss_one_q and loss_mix_q are losses from q.
        """
        # onehot q
        '''
        loss_one_q = None
        if self.head_one_q is not None and self.weight_one_q > 0:
            out_one_q = self.backbone_q(x)[-1]
            pred_one_q = self.head_one_q([out_one_q])
            # loss
            loss_one_q = self.head_one_q.loss(pred_one_q, y)
            if torch.isnan(loss_one_q["loss"]):
                print_log("Warming NAN in loss_one_q. Please use FP32!", logger='root')
                loss_one_q = None
        '''
        loss_one_q = dict(loss=None)
        pred_one_q = self.backbone_q(x)
        # loss
        loss_one_q['loss'] = self.criterion(pred_one_q, y).mean()
        if torch.isnan(loss_one_q["loss"]):
            loss_one_q['loss'] = None

        # mixup q
        '''
        loss_mix_q = None
        if self.weight_mix_q > 0:
            out_mix_q = self.backbone_q(mixed_x)[-1]
            pred_mix_q = self.head_mix_q([out_mix_q])
            # force fp32 in mixup loss (causing NAN in fp16 training with a large batch size)
            pred_mix_q[0] = pred_mix_q[0].type(torch.float32)
            # mixup loss
            y_mix_q = (y, y[index], lam)
            loss_mix_q = self.head_mix_q.loss(pred_mix_q, y_mix_q)
            if torch.isnan(loss_mix_q["loss"]):
                print_log("Warming NAN in loss_mix_q. Please use FP32!", logger='root')
                loss_mix_q = dict(loss=None)
        '''
        loss_mix_q = dict(loss=None)
        # force fp32 in mixup loss (causing NAN in fp16 training with a large batch size)
        pred_mix_q = self.backbone_q(mixed_x).type(torch.float32)
        # loss
        loss_mix_q['loss'] = self.criterion(pred_mix_q, y) * lam + self.criterion(pred_mix_q, y[index]) * (1. - lam)
        loss_mix_q['loss'] = loss_mix_q['loss'].mean()
        if torch.isnan(loss_mix_q["loss"]):
            loss_mix_q['loss'] = None

        return loss_one_q, loss_mix_q

    @torch.no_grad()
    def momentum_update(self):
        """Momentum update of the k form q by hook, including the backbone and heads """
        # we don't update q to k when momentum > 1
        if self.momentum >= 1.:
            return
        # update k's backbone and cls head from q
        for param_q, param_k in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
        
        '''
        if self.head_one_q is not None and self.head_one_k is not None:
            for param_one_q, param_one_k in zip(self.head_one_q.parameters(), self.head_one_k.parameters()):
                param_one_k.data = param_one_k.data * self.momentum + param_one_q.data * (1 - self.momentum)

        if self.head_mix_q is not None and self.head_mix_k is not None:
            for param_mix_q, param_mix_k in zip(self.head_mix_q.parameters(), self.head_mix_k.parameters()):
                param_mix_k.data = param_mix_k.data * self.momentum + param_mix_q.data * (1 - self.momentum)
        '''

    def mixing_info(self):
        return str(self.mixing_augmentation)

    def set_device(self, device):
        self.device = device
        self.mixing_augmentation.set_device(device)
