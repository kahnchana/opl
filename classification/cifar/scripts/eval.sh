#!/bin/bash

python train.py \
  -net resnet56 \
  -gpu \
  -b 256 \
  -eval \
  -resume \
  -pth "name_of_exp_folder. e.g: Thursday_01_April_2021_10h_10m_42s"
