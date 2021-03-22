#!/bin/bash

python train.py \
  -net resnet56 \
  -gpu \
  -lr 0.1 \
  -b 128 \
  -opl \
  -opl_ratio 1.0 \
  -opl_gamma 0.5
