#!/usr/bin/env bash
# Scallop on MNIST-Sum, top-3 proofs (difftopkproofs provenance).
set -euo pipefail
cd "$(dirname "$0")/mnist_sum"
PY="${PY:-$HOME/miniconda3/envs/casper_env/bin/python}"
NS="${NS:-2 16}"; SEEDS="${SEEDS:-1234 5678}"; EPOCHS="${EPOCHS:-100}"
LOGDIR="${LOGDIR:-../../../logs/baselines}"; mkdir -p "$LOGDIR"
for n in $NS; do for seed in $SEEDS; do
  echo ">>> Scallop sum n=$n seed=$seed"
  WANDB_MODE="${WANDB_MODE:-disabled}" "$PY" scallop_sum.py \
    --n-epochs "$EPOCHS" --digits "$n" --top-k 3 --seed "$seed" \
    2>&1 | tee "$LOGDIR/scallop_sum_n${n}_s${seed}.log"
done; done
