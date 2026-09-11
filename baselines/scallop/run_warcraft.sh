#!/usr/bin/env bash
# Scallop on Warcraft 12x12: per-cell categorical head (full alphabet), top-k
# proofs path loss in training, then the argmax and top-1 proof (MAP) decodes.
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
    --alphabet full 2>&1 | tee "$LOGDIR/scallop_warcraft_12_s${seed}.log"
  "$PY" baselines/scallop/warcraft/decode_map_warcraft.py --ckpt "$OUT/best_run0.pt" \
    --data_dir "$DATA" --out_json "$OUT/metrics_map_best_run0.json" --decoder both \
    --k 1 --alphabet full 2>&1 | tee "$LOGDIR/scallop_warcraft_12_s${seed}_map.log"
done
