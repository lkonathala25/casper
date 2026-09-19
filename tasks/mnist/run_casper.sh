#!/usr/bin/env bash
# CASPER on MNIST-Sum and MNIST-Add with the generated constraint files (checked
# to be functional-DAG fixpoints at load), layer in training and inference.
# Dataset sizes are fixed per task at every N, examples drawn with replacement so
# MNIST images are reused (no shrinking at large N): Add 6000/1000, Sum 5000/1000.
# 200 epochs, no early stopping, no time-out. Seeds 0-2 by default, one process
# per seed via launch_casper.sh; within a seed, settings run smallest N first.
# ADD_PROTOCOL=ctsketch instead deals each MNIST split into non-overlapping Add
# examples exactly as CTSketch's loader does (sizes shrink with N, e.g. 15000/2500
# at N=2, 585/97 at N=256); those runs go under add_ctsketch/.
# Env: TASKS, SUM_NS, ADD_NS, SEEDS, EPOCHS, ADD_TRAIN, ADD_TEST, SUM_TRAIN,
# SUM_TEST, ADD_PROTOCOL, RUNS_ROOT.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
SEEDS="${SEEDS:-0 1 2}"  # MNIST CASPER: 3 seeds
source tools/runlib.sh
TASKS="${TASKS:-sum add}"; SUM_NS="${SUM_NS:-2 4 16 64 256 1024}"; ADD_NS="${ADD_NS:-2 4 15 100 256}"
ADD_TRAIN="${ADD_TRAIN:-6000}"; ADD_TEST="${ADD_TEST:-1000}"; SUM_TRAIN="${SUM_TRAIN:-5000}"; SUM_TEST="${SUM_TEST:-1000}"
EPOCHS="${EPOCHS:-200}"; ADD_PROTOCOL="${ADD_PROTOCOL:-resample}"
for s in $SEEDS; do
  for task in $TASKS; do
    name=$task
    if [[ "$task" == sum ]]; then ns=$SUM_NS; data=(--num-train-samples "$SUM_TRAIN" --num-test-samples "$SUM_TEST")
    elif [[ "$ADD_PROTOCOL" == ctsketch ]]; then ns=$ADD_NS; data=(--data-protocol ctsketch); name=add_ctsketch
    else ns=$ADD_NS; data=(--num-train-samples "$ADD_TRAIN" --num-test-samples "$ADD_TEST"); fi
    for n in $ns; do
      dir="$RUNS_ROOT/mnist/casper/$name/n$n/seed$s"
      run_one "$dir" "$PY" tasks/mnist/train.py --task "$task" --n-digits "$n" --seed "$s" \
        --epochs "$EPOCHS" --out "$dir/results.json" --data-root data "${data[@]}"
    done
  done
done
finish_runs
