#!/bin/bash

# path to ImageNet validation set
IMAGENET=/path/to/ImageNet/val
# where to store the generated dataset
TARGET=/path/to/ImageNet-AdvBN
# vgg encoder
VGG=/path/to/prertained/vgg-19 network
# decoder
DECODER=/path/to/pretrained/decoder
# pgd arguments
STEP=0.2
EPS=1.1
REPEAT=6
# batchsize
BS=32

python make_test_data.py --sourcedir ${IMAGENET} --targetdir ${TARGET} --vgg ${VGG} --decoder ${DECODER}\
 --eps ${EPS} --nrepeat ${REPEAT} --step ${STEP} --batchsize ${BS}

