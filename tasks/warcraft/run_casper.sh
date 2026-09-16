#!/usr/bin/env bash
# CASPER on Warcraft: steps constraint loaded from the compiled artifact, full
# alphabet, layer in training and inference. Env: GRIDS, SEEDS, EPOCHS, RUNS_ROOT.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
source tools/runlib.sh
GRIDS="${GRIDS:-12}"; EPOCHS="${EPOCHS:-300}"
for g in $GRIDS; do for s in $SEEDS; do
  dir="$RUNS_ROOT/warcraft/${g}x${g}/casper/seed$s"
  run_one "$dir" "$PY" tasks/warcraft/train.py --data_dir "data/warcraft/${g}x${g}/preprocessed_4conn" \
    --grid-size "$g" --condition crl_in_training --constraint-mode steps --alphabet full \
    --n-runs 1 --seed "$s" --epochs "$EPOCHS" --num-workers 4 --save_dir "$dir"
done; done
finish_runs
