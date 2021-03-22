#!/bin/bash

python main.py \
  "path/to/imagenet/dataset" \
  -a resnet50 \
  -b 256 \
  -e \
  --resume "path/to/model"
