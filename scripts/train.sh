#!/bin/bash

# path to ImageNet
IMAGENET=/path/to/ImageNet/
# pgd arguments
STEP=0.2
EPS=1.1
REPEAT=6
# configuration
CONFIG=./configs/res50_configs.yml
# output prefix of the directory saving trained models
NAME=train_advbn

python train_imagenet.py ${IMAGENET} --output_prefix ${NAME} --config ${CONFIG} \
            --adv_step ${STEP} --eps ${EPS} --nrepeat ${REPEAT} \
