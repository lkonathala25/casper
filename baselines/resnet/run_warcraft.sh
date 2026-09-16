#!/usr/bin/env bash
# Unconstrained ResNet-18 on Warcraft: CASPER's trainer with no constraint layer,
# full alphabet. Env: GRIDS, SEEDS, EPOCHS, RUNS_ROOT.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
source tools/runlib.sh
GRIDS="${GRIDS:-12}"; EPOCHS="${EPOCHS:-300}"
for g in $GRIDS; do for s in $SEEDS; do
  dir="$RUNS_ROOT/warcraft/${g}x${g}/resnet/seed$s"
  run_one "$dir" "$PY" tasks/warcraft/train.py --data_dir "data/warcraft/${g}x${g}/preprocessed_4conn" \
    --grid-size "$g" --condition no_crl --constraint-mode none --alphabet full \
    --n-runs 1 --seed "$s" --epochs "$EPOCHS" --num-workers 4 --save_dir "$dir"
done; done
finish_runs
