#!/usr/bin/env bash
# DeepSoftLog on Warcraft: per-cell categorical head (full alphabet), steps path
# functor over the compiled domains. Runs from this directory (PYTHONPATH=.).
# Env: GRIDS, SEEDS, EPOCHS, RUNS_ROOT.
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
source "$ROOT/tools/runlib.sh"
GRIDS="${GRIDS:-12}"; EPOCHS="${EPOCHS:-100}"
for g in $GRIDS; do for s in $SEEDS; do
  out="$RUNS_ROOT/warcraft/${g}x${g}/deepsoftlog"
  run_one "$out/seed$s" env WANDB_MODE="${WANDB_MODE:-disabled}" PYTHONPATH=. "$PY" \
    deepsoftlog/experiments/warcraft/warcraft.py --config deepsoftlog/experiments/warcraft/config.yaml \
    --grid-size "$g" --data-dir "../../data/warcraft/${g}x${g}/preprocessed_4conn" \
    --seed "$s" --nb-epochs "$EPOCHS" --name "seed$s" --results-dir "$out"
done; done
finish_runs
