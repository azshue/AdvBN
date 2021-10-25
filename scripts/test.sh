#!/bin/bash

# path to ImageNet
IMAGENET=/path/to/ImageNet/val
# path to Stylized ImageNet
INS=/path/to/Stylized_ImageNet
# path to ImageNet-Sketch
INK=/path/to/ImageNet-Sketch
# path to ImageNet-C
INC=/path/to/ImageNet-C
# path to ImageNet-Ins
INT=/path/to/ImageNet-Instagram
# testset
sets='ISKCT'

# configuration
CONFIG=./configs/res50_configs.yml
# model checkpoint
CKPT=./path/to/pretrained/model

python ./test_imagenet.py -c ${CONFIG} --statedict ${CKPT} --set ${sets} \
        --pathI ${IMAGENET} --pathS ${INS} --pathK ${INK} \
        --pathC ${INC} --pathT ${INT}

