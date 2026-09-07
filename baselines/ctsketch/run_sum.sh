#!/usr/bin/env bash
# CTSketch on MNIST-Sum. n = 2**depth, so depth 1,4,6,8,10 -> n 2,16,64,256,1024.
set -euo pipefail
cd "$(dirname "$0")/mnist_sum"
PY="${PY:-$HOME/miniconda3/envs/casper_env/bin/python}"
DEPTHS="${DEPTHS:-1 4 6 8 10}"; SEEDS="${SEEDS:-1234 5678}"
EPOCHS="${EPOCHS:-100}"; LOGDIR="${LOGDIR:-../logs}"; mkdir -p "$LOGDIR"
for depth in $DEPTHS; do for seed in $SEEDS; do
  n=$((2**depth))
  echo ">>> CTSketch sum n=$n (depth=$depth) seed=$seed"
  WANDB_MODE="${WANDB_MODE:-disabled}" "$PY" cts_sum.py \
    --depth "$depth" --method tt --rank 2 --supervised --digit-loss-weight 1.0 \
    --n-epochs "$EPOCHS" --seed "$seed" 2>&1 | tee "$LOGDIR/cts_sum_n${n}_s${seed}.log"
done; done
