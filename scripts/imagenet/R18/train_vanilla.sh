#!/bin/bash

PORT="tcp://127.0.0.1:12345"
GPU=0
DATASET="imagenet"

SAVEDIR="saved/${DATASET}/R18"
NAME="vanilla"

python train.py -c configs/${DATASET}/resnet18/config_vanilla.json \
-d ${GPU} --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} --dataset ${DATASET}