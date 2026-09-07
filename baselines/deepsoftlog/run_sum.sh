#!/usr/bin/env bash
# DeepSoftLog on MNIST-Sum. Configs ship per n: config4/16/64/256/1024.yaml
# (and *_supervised variants). PYTHONPATH=. so the package resolves.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-$HOME/miniconda3/envs/casper_env/bin/python}"
CONFIGS="${CONFIGS:-config4.yaml config16_supervised.yaml}"
LOGDIR="${LOGDIR:-../../logs/baselines}"; mkdir -p "$LOGDIR"
for cfg in $CONFIGS; do
  echo ">>> DeepSoftLog sum $cfg"
  WANDB_MODE="${WANDB_MODE:-disabled}" PYTHONPATH=. "$PY" \
    deepsoftlog/run_experiments.py mnist_sum \
    "deepsoftlog/experiments/mnist_sum/$cfg" \
    2>&1 | tee "$LOGDIR/deepsoftlog_sum_${cfg%.yaml}.log"
done
