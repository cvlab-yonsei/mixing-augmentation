import torch
import numpy as np
import random
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

from pathlib import Path
from torch.utils.data import DataLoader
from data_loader import utils
from data_loader import DATASETS_IMG_DIRS


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ClassificationDataLoader():
    def __init__(self, dataset, batch_size, num_workers, pin_memory):
        self.dataset = dataset

        if dataset.startswith("cifar"):
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ])
            transform_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ])

            if dataset == "cifar10":
                self.n_class = 10
                base_dir = Path(DATASETS_IMG_DIRS[dataset])

                self.train_set = datasets.CIFAR10(
                    base_dir,
                    train=True,
                    download=True,
                    transform=transform_train,
                )

                self.val_set = datasets.CIFAR10(
                    base_dir,
                    train=False,
                    transform=transform_val,
                )

            elif dataset == "cifar100":
                self.n_class = 100
                base_dir = Path(DATASETS_IMG_DIRS[dataset])

                self.train_set = datasets.CIFAR100(
                    base_dir,
                    train=True,
                    download=True,
                    transform=transform_train,
                )

                self.val_set = datasets.CIFAR100(
                    base_dir,
                    train=False,
                    transform=transform_val,
                )
            else:
                raise Exception(f"unknown dataset: {dataset}")

        elif dataset == "tiny_imagenet":
            self.n_class = 200
            base_dir = Path(DATASETS_IMG_DIRS[dataset])

            self.train_set = datasets.ImageFolder(
                base_dir / "train",
                transforms.Compose([
                    transforms.RandomResizedCrop(64, interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
            )

            self.val_set = datasets.ImageFolder(
                base_dir / "val",
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
            )

            self.test_set = datasets.ImageFolder(
                base_dir / "test",
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
            )

        elif dataset == "imagenet":
            self.n_class = 1000
            base_dir = Path(DATASETS_IMG_DIRS[dataset])

            self.train_set = datasets.ImageFolder(
                base_dir / "train",
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # utils.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    # utils.Lighting(alphastd=0.1,
                    #                eigval=[0.2175, 0.0188, 0.0045],
                    #                eigvec=[[-0.5675, 0.7192, 0.4009],
                    #                        [-0.5808, -0.0045, -0.8140],
                    #                        [-0.5836, -0.6948, 0.4203]]),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
            )

            self.val_set = datasets.ImageFolder(
                base_dir / "val",
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
            )
        else:
            raise Exception(f"unknown dataset: {dataset}")
        
        self.init_kwargs = {'num_workers': num_workers, "pin_memory": pin_memory, "batch_size": batch_size}

    def get_train_loader(self, sampler=None):
        return DataLoader(self.train_set, **self.init_kwargs,
                          drop_last=True, sampler=sampler, shuffle=(sampler is None),)

    def get_val_loader(self, sampler=None):
        return DataLoader(self.val_set, **self.init_kwargs,
                          sampler=sampler, shuffle=False,)

    def __str__(self):
        return f"{self.dataset} DataLoader"

    def dataset_info(self):
        return f"The number of datasets: {len(self.train_set)} / {len(self.val_set)}"
