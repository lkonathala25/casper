#!/usr/bin/env bash
# Scallop on Warcraft 12x12: per-cell categorical head (full alphabet), steps walk
# program from the compiled artifact, top-3 proofs; evaluated by per-cell argmax.
# CPU-bound (proof search runs in scallopy on CPU): run seeds in parallel by
# launching one copy per seed, e.g. SEEDS=0 ... & SEEDS=1 ... &.
# 12x12 only: at 30x30 it exceeds the 5 s per-example time-out.
# Env: SEEDS, EPOCHS, K, RUNS_ROOT.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
source tools/runlib.sh
EPOCHS="${EPOCHS:-50}"; K="${K:-3}"
for s in $SEEDS; do
  dir="$RUNS_ROOT/warcraft/12x12/scallop/seed$s"
  run_one "$dir" "$PY" baselines/scallop/warcraft/scallop_warcraft.py \
    --data_dir data/warcraft/12x12/preprocessed_4conn --save_dir "$dir" --epochs "$EPOCHS" \
    --n-runs 1 --seed "$s" --k "$K" --alphabet full --constraint steps --resume
done
finish_runs
