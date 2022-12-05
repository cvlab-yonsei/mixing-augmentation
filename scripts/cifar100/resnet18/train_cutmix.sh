#!/bin/bash

PORT='tcp://127.0.0.1:12347'
GPU=0
SAVEDIR='saved'
NAME="gpu_1_cutmix"

DATASET='cifar100'

python train.py -c configs/cifar100/resnet18/config_cutmix.json \
-d ${GPU} --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--dataset ${DATASET}