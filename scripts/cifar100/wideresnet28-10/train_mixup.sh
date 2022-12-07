#!/bin/bash

PORT='tcp://127.0.0.1:12345'
GPU=0,1,2,3
SAVEDIR='saved'
NAME="gpu_4_mixup"

DATASET='cifar100'

python train.py -c configs/cifar100/wideresnet28-10/config_mixup.json \
-d ${GPU} --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--dataset ${DATASET}