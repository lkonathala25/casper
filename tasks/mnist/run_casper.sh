#!/usr/bin/env bash
# CASPER on MNIST-Sum and MNIST-Add with the generated constraint files (checked
# to be functional-DAG fixpoints at load), layer in training and inference.
# Dataset sizes are fixed per task at every N, examples drawn with replacement so
# MNIST images are reused (no shrinking at large N): Add 6000/1000, Sum 5000/1000.
# 200 epochs, no early stopping, no time-out. Seeds 0-2 by default, one process
# per seed via launch_casper.sh; within a seed, settings run smallest N first.
# Env: TASKS, SUM_NS, ADD_NS, SEEDS, EPOCHS, ADD_TRAIN, ADD_TEST, SUM_TRAIN,
# SUM_TEST, RUNS_ROOT.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
SEEDS="${SEEDS:-0 1 2}"  # MNIST CASPER: 3 seeds
source tools/runlib.sh
TASKS="${TASKS:-sum add}"; SUM_NS="${SUM_NS:-2 4 16 64 256 1024}"; ADD_NS="${ADD_NS:-2 4 15 100 256}"
ADD_TRAIN="${ADD_TRAIN:-6000}"; ADD_TEST="${ADD_TEST:-1000}"; SUM_TRAIN="${SUM_TRAIN:-5000}"; SUM_TEST="${SUM_TEST:-1000}"
EPOCHS="${EPOCHS:-200}"
for s in $SEEDS; do
  for task in $TASKS; do
    if [[ "$task" == sum ]]; then ns=$SUM_NS; ntr=$SUM_TRAIN; nte=$SUM_TEST
    else ns=$ADD_NS; ntr=$ADD_TRAIN; nte=$ADD_TEST; fi
    for n in $ns; do
      dir="$RUNS_ROOT/mnist/casper/$task/n$n/seed$s"
      run_one "$dir" "$PY" tasks/mnist/train.py --task "$task" --n-digits "$n" --seed "$s" \
        --epochs "$EPOCHS" --out "$dir/results.json" --data-root data \
        --num-train-samples "$ntr" --num-test-samples "$nte"
    done
  done
done
finish_runs
