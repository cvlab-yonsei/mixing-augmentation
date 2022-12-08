#!/bin/bash

PORT='tcp://127.0.0.1:12347'
GPU=0
SAVEDIR='saved'
NAME="gpu_1_cutmix_m"

DATASET='cifar100'

python train.py -c configs/cifar100/resnet18/config_resizemix_m.json \
-d ${GPU} --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--dataset ${DATASET} --ep 15 --save_dir "temp/"