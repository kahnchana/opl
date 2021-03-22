#!/bin/bash

DATA_PATH="path/to/imagenet"
PROJECT_PATH="path/to/repo/imagenet/dir"
EXP_NAME="exp_opl_01"

cd "$PROJECT_PATH" || exit

python main.py \
  "$DATA_PATH" \
  -a resnet50 \
  -b 1024 \
  --lr 0.01 \
  --opl \
  --save-path "$EXP_NAME" \
  --dist-url 'tcp://127.0.0.1:8091' \
  --dist-backend 'nccl' \
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0