#!/bin/bash

PORT="tcp://127.0.0.1:12345"
GPU=0,1
DATASET="imagenet"

SAVEDIR="saved/${DATASET}/R18"

NAME="puzzlemix_p_1.0"
python train.py -c configs/${DATASET}/resnet18/config_puzzlemix.json \
-d ${GPU} --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} --dataset ${DATASET}
