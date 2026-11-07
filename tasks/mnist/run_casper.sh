#!/usr/bin/env bash
# CASPER on MNIST-Sum and MNIST-Add with the generated constraint files (checked
# to be functional-DAG fixpoints at load). MODE and L2_PENALTY pick the CASPER
# configuration, with the same meaning as in the Warcraft scripts:
#   MODE=train_infer L2_PENALTY=1   full CASPER (task loss on the projected output + ½‖z̃−ŷ‖²)
#   MODE=train_infer L2_PENALTY=0   w/o l2 penalty
#   MODE=infer_only                 raw-logit task loss, layer only at test
#   MODE=train_only L2_PENALTY=1    layer only in training (Train-Only is normally read
#                                   off the full-CASPER checkpoints instead)
# Runs go under casper/<task>_<variant>/, the variant built by casper_variant
# (tools/runlib.sh): sum_ctsketch_<variant>/ and add_ctsketch_<variant>/, e.g.
# sum_ctsketch_train_infer_l2_penalty1/.
# Both tasks use CTSketch's data construction, the only one (tasks/mnist/data.py):
# MNIST images dealt into non-overlapping examples. Sum: 5000 train (of 6000
# dealt) / 1000 test at every N. Add: each split repeated 1-5x by N, 15000/2500 at
# N=2 down to 585/97 at N=256.
# 200 epochs, no early stopping, no time-out. Seeds 0-4 by default, one process
# per seed via launch_casper.sh; within a seed, settings run smallest N first.
# BACKEND=cvxpy solves the same projection as a QP (SCS) instead of the closed
# form. TIMEOUT=5 applies the paper's 5 s per training example: a step over
# budget ends the run with status timed_out (reported as T/O).
# RUNS_SUBDIR moves the run folders under $RUNS_ROOT (default mnist/casper); the
# idle-GPU timing trials use mnist/timing/casper.
# Env: MODE (required), L2_PENALTY, TASKS, SUM_NS, ADD_NS, SEEDS, EPOCHS, BACKEND,
# TIMEOUT, RUNS_ROOT, RUNS_SUBDIR.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
SEEDS="${SEEDS:-0 1 2 3 4}"
source tools/runlib.sh
TASKS="${TASKS:-sum add}"; SUM_NS="${SUM_NS:-2 16 64 256 1024}"; ADD_NS="${ADD_NS:-1 2 4 15 100 256}"
EPOCHS="${EPOCHS:-200}"
BACKEND="${BACKEND:-closed_form}"; TIMEOUT="${TIMEOUT:-}"
: "${MODE:?set MODE=train_infer|train_only|infer_only|none (and L2_PENALTY for the train modes)}"
L2_PENALTY="${L2_PENALTY:-}"
variant="$(casper_variant "$MODE" "$L2_PENALTY" "$BACKEND")" || exit 1
# Every backend runs in the batched layer, so cvxpy and the closed form share the
# same batching and differ only in how each projection is solved.
extra=(--projection-backend "$BACKEND" --casper-mode "$MODE")
[[ -n "$L2_PENALTY" ]] && extra+=(--l2-penalty "$L2_PENALTY")
[[ -n "$TIMEOUT" ]] && extra+=(--per-example-timeout-s "$TIMEOUT")
for s in $SEEDS; do
  for task in $TASKS; do
    if [[ "$task" == sum ]]; then ns=$SUM_NS; else ns=$ADD_NS; fi
    for n in $ns; do
      dir="$RUNS_ROOT/${RUNS_SUBDIR:-mnist/casper}/${task}_ctsketch_$variant/n$n/seed$s"
      run_one "$dir" "$PY" tasks/mnist/train.py --task "$task" --n-digits "$n" --seed "$s" \
        --epochs "$EPOCHS" --out "$dir/results.json" --data-root data "${extra[@]}"
    done
  done
done
finish_runs
