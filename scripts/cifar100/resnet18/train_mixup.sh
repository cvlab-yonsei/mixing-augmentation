#!/bin/bash

PORT='tcp://127.0.0.1:12346'
GPU=1
SAVEDIR='saved'
NAME="gpu_1_mixup"

DATASET='cifar100'

python train.py -c configs/cifar100/resnet18/config_mixup.json \
-d ${GPU} --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--dataset $