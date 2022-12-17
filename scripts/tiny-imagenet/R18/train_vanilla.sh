#!/bin/bash

PORT='tcp://127.0.0.1:12345'
GPU=0
DATASET='tiny_imagenet'

SAVEDIR='saved/tiny_imagenet/R18'
NAME="vanilla"

python train.py -c configs/tiny_imagenet/resnet18/config_vanilla.json \
-d ${GPU} --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--dataset ${DATASET} --bs 100 --ep 400