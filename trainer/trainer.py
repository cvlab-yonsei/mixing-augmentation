import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import utils.mixing as module_mixing

from base import BaseTrainer
from utils import MetricTracker, MetricTracker_scalars
from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer(BaseTrainer):
    def __init__(
        self, model, optimizer, evaluator, config,
        data_loader, lr_scheduler=None, logger=None, gpu=None
    ):
        super().__init__(config, logger, gpu)
        self.optimizer = optimizer
        self.train_evaluator = evaluator[0]
        self.valid_evaluator = evaluator[1]

        self.train_loader = data_loader[0]
        self.len_epoch = len(self.train_loader)

        self.valid_loader = data_loader[1]

        self.lr_scheduler = lr_scheduler

        # For automatic mixed precision(AMP)
        if config['use_amp'] is True:
            self.scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp'])
        else:
            self.scaler = None

        if self.train_evaluator is not None:
            self.train_metric_ftns = [getattr(self.train_evaluator, met) for met in config['metrics']]
        if self.valid_evaluator is not None:
            self.valid_metric_ftns = [getattr(self.valid_evaluator, met) for met in config['metrics']]

        self.train_metrics = MetricTracker(
            'loss',
            writer=self.writer,
            colums=['total', 'counts', 'average'],
        )
        self.valid_metrics = MetricTracker_scalars(writer=self.writer)

        if config.resume is not None:
            self._resume_checkpoint(config.resume, config['test'])
        
        if not torch.cuda.is_available():
            self.logger.info("using CPU, this will be slow")

        elif config['multiprocessing_distributed']:
            if gpu is not None:
                torch.cuda.set_device(self.device)
                model.to(self.device)
                # When using a single GPU per process and per
                # DDP, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False

            else:
                model.to(self.device)
                # DDP will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                self.model = DDP(model)

        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            self.model = nn.DataParallel(model, device_ids=self.device_ids)
            self.model.to(self.device)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        torch.distributed.barrier()
        self.model.train()
        self.train_metrics.reset()
        self.logger.info(f'Epoch - {epoch}')

        # Random shuffling
        if not isinstance(self.train_loader.sampler, torch.utils.data.RandomSampler):
            self.train_loader.sampler.set_epoch(epoch)
        for batch_idx, (image, target) in enumerate(self.train_loader):
            image, target = image.to(self.device), target.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                output = self.model(image, target)
                loss = output['total_loss'].mean()
            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            # Get First lr
            if batch_idx == 0:
                self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch - 1)
                self.logger.info(f"lr: {self.optimizer.param_groups[0]['lr']:.6f}")

            self.progress(self.logger, batch_idx, len(self.train_loader))

            if batch_idx == self.len_epoch:
                break

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # average train loss per epoch
        log = self.train_metrics.result()

        return log

    def _validation(self, epoch=None):
        torch.distributed.barrier()
        self.model.eval()
        log = {}
        self.valid_evaluator.reset()

        with torch.no_grad():
            for batch_idx, (image, target) in enumerate(self.valid_loader):
                image, target = image.to(self.device), target.to(self.device)
                
                output = self.model(image, inference=True)
                acc1, acc5 = self._accuracy(output, target, topk=(1, 5))
                self.valid_evaluator.update((acc1.item(), acc5.item()), image.size(0))

            if epoch is not None:
                if self.rank == 0:
                    self.writer.set_step((epoch), 'val')

            for met in self.valid_metric_ftns:
                if epoch is not None:
                    self.valid_metrics.update(met.__name__, [met()['top1'], met()['top5']], 'top1', 'top5', n=1)

                if 'top1' in met().keys():
                    log.update({met.__name__ + '_top1': met()['top1']})
                if 'top5' in met().keys():
                    log.update({met.__name__ + '_top5': met()['top5']})

        return log

    def _test(self, epoch=None):
        torch.distributed.barrier()
        self.model.eval()
        log = {}
        self.valid_evaluator.reset()

        with torch.no_grad():
            for batch_idx, (image, target) in enumerate(self.val_loader):
                image, target = image.to(self.device), target.to(self.device)
                
                output = self.model(image, inference=True)
                acc1, acc5 = self._accuracy(output, target, topk=(1, 5))
                self.valid_evaluator.update((acc1.item(), acc5.item()), image.size(0))

            if epoch is not None:
                if self.rank == 0:
                    self.writer.set_step((epoch), 'test')

            for met in self.valid_metric_ftns:
                if epoch is not None:
                    self.valid_metrics.update(met.__name__, [met()['top1'], met()['top5']], 'top1', 'top5', n=1)

                if 'top1' in met().keys():
                    log.update({met.__name__ + '_top1': met()['top1']})
                if 'top5' in met().keys():
                    log.update({met.__name__ + '_top5': met()['top5']})

        return log

    def _accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # wrong_k = batch_size - correct_k
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
