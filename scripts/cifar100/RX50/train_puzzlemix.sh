#!/bin/bash

PORT='tcp://127.0.0.1:12347'
GPU=0
SAVEDIR='saved'
NAME="gpu_1_puzzlemix"

DATASET='cifar100'

python train.py -c configs/cifar100/resnext50-32x4d/config_puzzlemix.json \
-d ${GPU} --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--dataset ${DATASET}