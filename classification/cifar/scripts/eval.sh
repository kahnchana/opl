#!/bin/bash

python train.py \
  -net resnet56 \
  -gpu \
  -b 256 \
  -eval \
  -resume \
  -pth "path/to/model/dir"
