#!/usr/bin/env bash
# CTSketch on MNIST-Add. --digits is the operand width.
set -euo pipefail
cd "$(dirname "$0")/mnist_add"
PY="${PY:-$HOME/miniconda3/envs/casper_env/bin/python}"
DIGITS="${DIGITS:-2 4 15 100 256}"; SEEDS="${SEEDS:-1234 5678}"
EPOCHS="${EPOCHS:-100}"; LOGDIR="${LOGDIR:-../logs}"; mkdir -p "$LOGDIR"
for n in $DIGITS; do for seed in $SEEDS; do
  echo ">>> CTSketch add digits=$n seed=$seed"
  WANDB_MODE="${WANDB_MODE:-disabled}" "$PY" cts_add.py \
    --digits "$n" --method tt --supervised --digit-loss-weight 1.0 \
    --n-epochs "$EPOCHS" --batch-size 64 --learning-rate 1e-3 --seed "$seed" \
    2>&1 | tee "$LOGDIR/cts_add_n${n}_s${seed}.log"
done; done
