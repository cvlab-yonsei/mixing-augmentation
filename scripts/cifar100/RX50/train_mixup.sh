#!/bin/bash

PORT="tcp://127.0.0.1:12345"
GPU=0
DATASET="cifar100"

SAVEDIR="saved/${DATASET}/RX50"
NAME="mixup"

python train.py -c configs/${DATASET}/resnext50-32x4d/config_mixup.json \
-d ${GPU} --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} --dataset ${DATASET}