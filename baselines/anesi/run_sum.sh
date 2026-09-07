#!/usr/bin/env bash
# A-NeSI on MNIST-Sum: N=1, arity = number of digits.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-$HOME/miniconda3/envs/casper_env/bin/python}"
NS="${NS:-2 16 64 256 1024}"; SEEDS="${SEEDS:-1234 5678}"
EPOCHS="${EPOCHS:-50}"; BS="${BS:-16}"; LOGDIR="${LOGDIR:-logs}"; mkdir -p "$LOGDIR"
for n in $NS; do for seed in $SEEDS; do
  CFG="$LOGDIR/_cfg_sum_n${n}_s${seed}.yaml"
  cat > "$CFG" <<CFGEOF
N: 1
arity: $n
supervised: true
digit_loss_weight: 1.0
epochs: $EPOCHS
seed: $seed
batch_size: $BS
batch_size_test: $BS
hidden_size: 200
CFGEOF
  echo ">>> A-NeSI sum n=$n seed=$seed"
  WANDB_MODE="${WANDB_MODE:-disabled}" "$PY" -m experiments.mnist_op.mnistop \
    --config "$CFG" 2>&1 | tee "$LOGDIR/anesi_sum_n${n}_s${seed}.log"
done; done
