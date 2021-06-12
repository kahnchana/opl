# supervised pre-training

DATASET="CIFAR-FS"
#DATASET="miniImageNet"
#DATASET="tieredImageNet"
DATA_ROOT="$HOME/data/cache"
EXP="opl_01"

python train_supervised.py \
  --dataset "${DATASET}" \
  --batch_size 128 \
  --trial "${EXP}" \
  --model_path ckpts \
  --tb_path ckpts \
  --popl \
  --opl_ratio 0.1 \
  --data_root "${DATA_ROOT}" 2>&1
