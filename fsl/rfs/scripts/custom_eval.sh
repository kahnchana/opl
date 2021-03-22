# supervised pre-training evaluation

MODEL=$1
DATASET="CIFAR-FS"
#DATASET="miniImageNet"
#DATASET="tieredImageNet"
DATA_ROOT="$HOME/data/cache/$DATASET"
PROJECT_PATH="$HOME/repo/rfs"

cd "$PROJECT_PATH" || exit

python eval_fewshot.py \
  --dataset "$DATASET" \
  --model_path "$MODEL" \
  --n_shots 1 \
  --n_aug_support_samples 5 \
  --n_test_runs 3000 \
  --data_root "$DATA_ROOT"

python eval_fewshot.py \
  --dataset "$DATASET" \
  --model_path "$MODEL" \
  --n_shots 5 \
  --n_aug_support_samples 1 \
  --n_test_runs 3000 \
  --data_root "$DATA_ROOT"