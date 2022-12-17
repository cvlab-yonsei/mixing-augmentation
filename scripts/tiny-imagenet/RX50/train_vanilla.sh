#!/bin/bash

PORT='tcp://127.0.0.1:12346'
GPU=1,2
DATASET='tiny_imagenet'

SAVEDIR='saved/tiny_imagenet/RX50'
NAME="vanilla"

python train.py -c configs/tiny_imagenet/resnext50-32x4d/config_vanilla.json \
-d ${GPU} --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--dataset ${DATASET} --bs 100 --ep 400