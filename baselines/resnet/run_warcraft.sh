#!/usr/bin/env bash
# Unconstrained ResNet-18 on Warcraft: CASPER's own trainer with no constraint
# layer (--condition no_crl --constraint-mode none), full per-cell alphabet.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
PY="${PY:-$HOME/miniconda3/envs/casper_env/bin/python}"
GRIDS="${GRIDS:-12}"; RUNS="${RUNS:-3}"; EPOCHS="${EPOCHS:-300}"; SEED="${SEED:-42}"
LOGDIR="${LOGDIR:-logs/baselines}"; mkdir -p "$LOGDIR"
for g in $GRIDS; do
  echo ">>> ResNet (unconstrained) warcraft ${g}x${g}"
  "$PY" tasks/warcraft/train.py --data_dir "data/warcraft/${g}x${g}/preprocessed_4conn" \
    --grid-size "$g" --condition no_crl --constraint-mode none --alphabet full \
    --n-runs "$RUNS" --epochs "$EPOCHS" --seed "$SEED" \
    --save_dir "baselines/resnet/warcraft/results/${g}x${g}" \
    2>&1 | tee "$LOGDIR/resnet_warcraft_${g}.log"
done
