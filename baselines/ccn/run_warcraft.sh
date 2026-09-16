#!/usr/bin/env bash
# CCN+ (pishield) on Warcraft: per-cell categorical head (full alphabet), shield
# over the steps atoms, compiled once into constraints/warcraft/ccn/.
# Env: GRIDS, SEEDS, EPOCHS, RUNS_ROOT.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
source tools/runlib.sh
GRIDS="${GRIDS:-12}"; EPOCHS="${EPOCHS:-300}"
for g in $GRIDS; do
  if [[ ! -f "constraints/warcraft/ccn/warcraft_ccn_steps_${g}x${g}_strata.pkl" ]]; then
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
      echo "[dry-run] would compile the CCN+ steps shield for ${g}x${g} first"
    else
      echo ">>> compiling CCN+ steps shield ${g}x${g} (once)"
      "$PY" baselines/ccn/warcraft/compile_steps_shield.py --grid-size "$g" 2>&1 \
        | grep -E "^requirements|^RESULT|Error"
    fi
  fi
  for s in $SEEDS; do
    dir="$RUNS_ROOT/warcraft/${g}x${g}/ccn/seed$s"
    run_one "$dir" "$PY" baselines/ccn/warcraft/train_ccn.py \
      --data_dir "data/warcraft/${g}x${g}/preprocessed_4conn" --grid-size "$g" --alphabet full \
      --constraint steps --n-runs 1 --epochs "$EPOCHS" --seed "$s" --num-workers 4 --save_dir "$dir"
  done
done
finish_runs
