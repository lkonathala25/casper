#!/usr/bin/env bash
# SPL on Warcraft 12x12 over the steps encoding (StepsFFCircuit), SPL's own
# main/trainers unmodified. Selects the epoch on validation with SPL's own
# criterion, then scores the test split once with the shared paper_metrics.
# Runs seeds sequentially; for parallel seeds launch one copy per GPU, e.g.
# CUDA_VISIBLE_DEVICES=0 SEEDS=0 ... & CUDA_VISIBLE_DEVICES=1 SEEDS=1 ... &.
# 12x12 only: SPL's path circuit does not compile at 30x30 (T/O).
# Env: SEEDS, RUNS_ROOT.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
source tools/runlib.sh
CFG="settings/warcraft_shortest_path/spl/12x12_steps.json"
for s in $SEEDS; do
  dir="$RUNS_ROOT/warcraft/12x12/spl/seed$s"
  PARAMS="{'default_json': '$CFG', 'seed': $s, 'model_dir': '$dir'}"
  run_one "$dir" bash -c 'cd baselines/spl/warcraft && "$0" spl_steps_main.py "$1" && "$0" spl_steps_test.py "$2"' \
    "$PY" "$PARAMS" "$dir"
done
finish_runs
