# supervised pre-training

DATASET="CIFAR-FS"
#DATASET="miniImageNet"
#DATASET="tieredImageNet"
DATA_ROOT="$HOME/data/cache"

train() {
python train_supervised.py \
  --dataset "${DATASET}" \
  --batch_size 128 \
  --trial "$3" \
  --model_path ckpts \
  --tb_path ckpts \
  --opl \
  --opl_ratio "$1" \
  --opl_gamma "$2" \
  --data_root "${DATA_ROOT}" \
  --save_freq 100
}

for lambda in 0.05 0.1 1.0 2.0 ; do
  for gamma in 0.5 1.0 2.0; do
      train "$lambda" "$gamma" "exp_${lambda}_${gamma}"
  done
done
