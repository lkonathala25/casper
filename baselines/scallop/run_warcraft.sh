#!/usr/bin/env bash
# Scallop on Warcraft 12x12: per-cell categorical head (full alphabet), steps-program
# top-k proofs path loss in training, per-cell argmax at inference.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
PY="${PY:-$HOME/miniconda3/envs/casper_env/bin/python}"
SEEDS="${SEEDS:-42 43 44}"; EPOCHS="${EPOCHS:-50}"; K="${K:-3}"
DATA="data/warcraft/12x12/preprocessed_4conn"
LOGDIR="${LOGDIR:-logs/baselines}"; mkdir -p "$LOGDIR"
for seed in $SEEDS; do
  OUT="baselines/scallop/warcraft/results/seed${seed}"
  echo ">>> Scallop warcraft 12x12 seed=$seed k=$K"
  "$PY" baselines/scallop/warcraft/scallop_warcraft.py --data_dir "$DATA" \
    --save_dir "$OUT" --epochs "$EPOCHS" --n-runs 1 --seed "$seed" --k "$K" \
    --alphabet full --constraint steps 2>&1 | tee "$LOGDIR/scallop_warcraft_12_s${seed}.log"
done
