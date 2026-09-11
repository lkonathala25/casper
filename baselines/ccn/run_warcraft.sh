#!/usr/bin/env bash
# CCN+ (pishield) on Warcraft: per-cell categorical head (full alphabet), shield over
# the steps atoms X_i = v, compiled once into constraints/warcraft/ccn/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
PY="${PY:-$HOME/miniconda3/envs/casper_env/bin/python}"
GRIDS="${GRIDS:-12}"; RUNS="${RUNS:-2}"; EPOCHS="${EPOCHS:-300}"; SEED="${SEED:-42}"
LOGDIR="${LOGDIR:-logs/baselines}"; mkdir -p "$LOGDIR"
for g in $GRIDS; do
  if [ ! -f "constraints/warcraft/ccn/warcraft_ccn_steps_${g}x${g}_strata.pkl" ]; then
    echo ">>> compiling CCN+ steps shield ${g}x${g} (once)"
    "$PY" baselines/ccn/warcraft/compile_steps_shield.py --grid-size "$g" 2>&1 \
      | grep -E "^requirements|^RESULT|Error" | tee "$LOGDIR/ccn_compile_${g}.log"
  fi
  echo ">>> CCN+ warcraft ${g}x${g}"
  "$PY" baselines/ccn/warcraft/train_ccn.py --data_dir "data/warcraft/${g}x${g}/preprocessed_4conn" \
    --grid-size "$g" --alphabet full --constraint steps --n-runs "$RUNS" --epochs "$EPOCHS" \
    --seed "$SEED" --save_dir "baselines/ccn/warcraft/results/${g}x${g}" \
    2>&1 | tee "$LOGDIR/ccn_warcraft_${g}.log"
done
