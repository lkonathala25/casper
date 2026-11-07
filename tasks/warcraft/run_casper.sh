#!/usr/bin/env bash
# CASPER on Warcraft: steps constraint loaded from the compiled artifact, full
# alphabet. MODE and L2_PENALTY pick the CASPER configuration, with the same
# meaning as in tasks/mnist/run_casper.sh:
#   MODE=train_infer L2_PENALTY=2   full CASPER as reported (task loss on the projected output + λ·½‖z̃−ŷ‖²)
#   MODE=train_infer L2_PENALTY=0   w/o l2 penalty
#   MODE=infer_only                 raw-logit task loss, layer only at test
#   MODE=train_only L2_PENALTY=1    layer only in training
# Runs go under <N>x<N>/casper_<variant>/seed<S>/, the variant built by
# casper_variant (tools/runlib.sh), e.g. casper_train_infer_l2_penalty1/.
# Early stopping (patience 30) cannot end a run before MIN_EPOCHS (100): with the
# layer at validation an untrained network already scores near its ceiling at
# 30x30, so the first epochs can hold the best score by chance.
# Env: MODE (required), L2_PENALTY, GRIDS, SEEDS, EPOCHS, MIN_EPOCHS, RUNS_ROOT.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
source tools/runlib.sh
GRIDS="${GRIDS:-12}"; EPOCHS="${EPOCHS:-300}"; MIN_EPOCHS="${MIN_EPOCHS:-100}"
: "${MODE:?set MODE=train_infer|train_only|infer_only (and L2_PENALTY for the train modes)}"
L2_PENALTY="${L2_PENALTY:-}"
variant="$(casper_variant "$MODE" "$L2_PENALTY")" || exit 1
extra=(--casper-mode "$MODE")
[[ -n "$L2_PENALTY" ]] && extra+=(--l2-penalty "$L2_PENALTY")
for g in $GRIDS; do for s in $SEEDS; do
  dir="$RUNS_ROOT/warcraft/${g}x${g}/casper_$variant/seed$s"
  run_one "$dir" "$PY" tasks/warcraft/train.py --data_dir "data/warcraft/${g}x${g}/preprocessed_4conn" \
    --grid-size "$g" --constraint-mode steps --alphabet full "${extra[@]}" \
    --n-runs 1 --seed "$s" --epochs "$EPOCHS" --min-epochs "$MIN_EPOCHS" --num-workers 4 --save_dir "$dir"
done; done
finish_runs
