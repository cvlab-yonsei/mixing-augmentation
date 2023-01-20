import time
import datetime
import argparse
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

import models.model as module_arch
import utils.metric as module_metric
import utils.lr_scheduler as module_lr_scheduler
import data_loader.data_loaders as module_data
from trainer.trainer import Trainer
from utils.parse_config import ConfigParser
from logger.logger import Logger

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True


def main(config):
    ngpus_per_node = torch.cuda.device_count()
    if config['multiprocessing_distributed']:
        # Single node, mutliple GPUs
        config.config['world_size'] = ngpus_per_node * config['world_size']
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Rather using distributed, use DataParallel
        main_worker(None, ngpus_per_node, config)


def main_worker(gpu, ngpus_per_node, config):
    if config['multiprocessing_distributed']:
        config.config['rank'] = config['rank'] * ngpus_per_node + gpu

    dist.init_process_group(
        backend=config['dist_backend'], init_method=config['dist_url'],
        world_size=config['world_size'], rank=config['rank']
    )
    
    # Set looging
    rank = dist.get_rank()
    logger = Logger(config.log_dir, rank=rank)
    logger.set_logger(f'train(rank{rank})', verbosity=2)

    # fix random seeds for reproduce
    SEED = config['seed']
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # 추가
    np.random.seed(SEED)
    random.seed(SEED)

    # Create Dataloader
    dataset = config.init_obj('data_loader', module_data)
    logger.info(f"{str(dataset)}")
    logger.info(f"{dataset.dataset_info()}")

    if config['multiprocessing_distributed']:
        train_sampler = DistributedSampler(dataset.train_set)
    else:
        train_sampler = None

    train_loader = dataset.get_train_loader(train_sampler)
    val_loader = dataset.get_val_loader()
    
    # Create Model
    model = config.init_obj('arch', module_arch, **{"num_classes": dataset.n_class, "config": config})

    # Convert BN to SyncBN
    if config['multiprocessing_distributed']:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    logger.info(model)
    logger.info('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # Build optimizer
    optimizer = config.init_obj(
        'optimizer',
        torch.optim,
        [{"params": filter(lambda p: p.requires_grad, model.parameters())}]
    )

    lr_scheduler = config.init_obj(
        'lr_scheduler',
        module_lr_scheduler,
        **{"optimizer": optimizer,
           "num_epochs": config["trainer"]['epochs'],
           "iters_per_epoch": len(train_loader)}
    )

    evaluator_train = config.init_obj(
        'evaluator',
        module_metric,
    )

    evaluator_val = config.init_obj(
        'evaluator',
        module_metric,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        evaluator=(evaluator_train, evaluator_val),
        config=config,
        data_loader=(train_loader, val_loader),
        lr_scheduler=lr_scheduler,
        logger=logger, gpu=gpu,
    )

    logger.print(f"{torch.randint(0, 100, (1, 1))}")
    torch.distributed.barrier()

    if config['test'] is not True:
        start = time.time()
        trainer.train()
        runtime = datetime.timedelta(seconds=(time.time() - start))
        logger.info(f"Runtime: {runtime}")
    else:
        trainer.test()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Mixing Augmentations')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type action target', defaults=(None, float, None, None))
    options = [
        CustomArgs(['--multiprocessing_distributed'], action='store_true', target='multiprocessing_distributed'),
        CustomArgs(['--dist_url'], type=str, target='dist_url'),

        CustomArgs(['--name'], type=str, target='name'),
        CustomArgs(['--test'], action='store_true', target='test'),

        CustomArgs(['--seed'], type=int, target='seed'),
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--wd', '--weight_decay'], type=float, target='optimizer;args;weight_decay'),
        
        CustomArgs(['--dataset'], type=str, target='data_loader;args;dataset'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),

        CustomArgs(['--ep', '--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--save_ep'], type=int, target='trainer;save_period'),
        
        CustomArgs(['--save_dir'], type=str, target='trainer;save_dir'),

        CustomArgs(['--arch'], type=str, target='arch;type'),
        
        CustomArgs(['--mixing'], type=str, target='mixing_augmentation;type'),
        CustomArgs(['--distribution'], type=str, target='mixing_augmentation;args;distribution'),
        CustomArgs(['--alpha1'], type=float, target='mixing_augmentation;args;alpha1'),
        CustomArgs(['--alpha2'], type=float, target='mixing_augmentation;args;alpha2'),
        CustomArgs(['--mix_prob'], type=float, target='mixing_augmentation;args;mix_prob'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
