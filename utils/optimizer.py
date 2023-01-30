import re
import torch
from typing import Dict, Optional


class DefaultOptimizer():
    def __init__(
        self,
        optimizer_cfg: Dict,
        paramwise_cfg: Optional[Dict] = None
    ):
        self.optimizer_cfg = optimizer_cfg
        self.paramwise_cfg = None if paramwise_cfg is None else paramwise_cfg
        self.base_lr = optimizer_cfg.get('lr', None)
        self.base_wd = optimizer_cfg.get('weight_decay', None)

    def __call__(self, model):
        """ add `paramwise_cfg` option for `DefaultOptimizer` """
        if hasattr(model, 'module'):
            model = model.module
        optimizer_cfg = self.optimizer_cfg.copy()
        paramwise_options = self.paramwise_cfg

        optimizer_type = optimizer_cfg.pop("type")

        # if no paramwise option is specified, just use the global setting
        if paramwise_options is None:
            optimizer_cfg['params'] = model.parameters()
            return getattr(torch.optim, optimizer_type)(**optimizer_cfg)
        else:
            assert isinstance(paramwise_options, dict)
            params = []
            for name, param in model.named_parameters():
                param_group = {'params': [param]}
                if not param.requires_grad:
                    params.append(param_group)
                    continue
                for regexp, options in paramwise_options.items():
                    if re.search(regexp, name):
                        for key, value in options.items():
                            param_group[key] = value

                # otherwise use the global settings
                params.append(param_group)
            optimizer_cfg['params'] = params

            return getattr(torch.optim, optimizer_type)(**optimizer_cfg)
