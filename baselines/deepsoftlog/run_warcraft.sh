#!/usr/bin/env bash
# DeepSoftLog on Warcraft 12x12 (per-cell categorical head, full alphabet).
# PYTHONPATH=. so the package resolves; results land in results/<name>/.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-$HOME/miniconda3/envs/casper_env/bin/python}"
SEEDS="${SEEDS:-42 43}"; EPOCHS="${EPOCHS:-100}"; GRID="${GRID:-12}"
LOGDIR="${LOGDIR:-../../logs/baselines}"; mkdir -p "$LOGDIR"
for seed in $SEEDS; do
  echo ">>> DeepSoftLog warcraft ${GRID}x${GRID} seed=$seed"
  WANDB_MODE="${WANDB_MODE:-disabled}" PYTHONPATH=. "$PY" \
    deepsoftlog/experiments/warcraft/warcraft.py \
    --config deepsoftlog/experiments/warcraft/config.yaml \
    --grid-size "$GRID" --data-dir "../../data/warcraft/${GRID}x${GRID}/preprocessed_4conn" \
    --seed "$seed" --nb-epochs "$EPOCHS" --name "warcraft-${GRID}-max-s${seed}" \
    2>&1 | tee "$LOGDIR/deepsoftlog_warcraft_${GRID}_s${seed}.log"
done
