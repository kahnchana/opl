# supervised pre-training
EXP="opl_01"
DATASET="CIFAR-FS"
#DATASET="miniImageNet"
#DATASET="tieredImageNet"
DATA_ROOT="$HOME/data/cache"
PROJECT_PATH="$HOME/repo/rfs"

cd "$PROJECT_PATH" || exit

python train_supervised.py \
  --dataset "${DATASET}" \
  --batch_size 128 \
  --trial "${EXP}" \
  --model_path ckpts \
  --tb_path ckpts \
  --popl \
  --opl_ratio 0.1 \
  --data_root "${DATA_ROOT}" 2>&1
