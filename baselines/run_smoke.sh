#!/usr/bin/env bash
# One-epoch smoke test for a ported baseline, at the smallest n.
# Proves the port runs end to end; says nothing about accuracy.
#
#   ./run_smoke.sh <method> <task>
#   method: ctsketch | anesi | scallop | deepproblog | deepsoftlog | spl
#   task:   sum | add
#
# Logs to logs/smoke/<method>_<task>.log and appends a one-line verdict.
set -uo pipefail
METHOD="${1:?method}"; TASK="${2:?task}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$HOME/miniconda3/envs/casper_env/bin/python}"
LOG="$ROOT/logs/smoke/${METHOD}_${TASK}.log"
mkdir -p "$(dirname "$LOG")"
export WANDB_MODE=disabled
: > "$LOG"
echo "# $METHOD $TASK  started $(date -Is)" >> "$LOG"
t0=$SECONDS

case "$METHOD:$TASK" in
  ctsketch:sum)  cd "$ROOT/baselines/ctsketch/mnist_sum"
    "$PY" cts_sum.py --depth 1 --method tt --rank 2 --supervised \
      --digit-loss-weight 1.0 --n-epochs 1 --seed 1234 >>"$LOG" 2>&1 ;;
  ctsketch:add)  cd "$ROOT/baselines/ctsketch/mnist_add"
    "$PY" cts_add.py --digits 2 --method tt --supervised --digit-loss-weight 1.0 \
      --n-epochs 1 --batch-size 64 --learning-rate 1e-3 --seed 1234 >>"$LOG" 2>&1 ;;
  anesi:sum|anesi:add)  cd "$ROOT/baselines/anesi"
    CFG="$ROOT/logs/smoke/_cfg_anesi_${TASK}.yaml"
    if [ "$TASK" = sum ]; then printf 'N: 1\narity: 2\n' > "$CFG"
    else printf 'N: 2\narity: 2\nop: add_n\n' > "$CFG"; fi
    printf 'supervised: true\ndigit_loss_weight: 1.0\nepochs: 1\nseed: 1234\nbatch_size: 16\nbatch_size_test: 16\nhidden_size: 200\n' >> "$CFG"
    "$PY" -m experiments.mnist_op.mnistop --config "$CFG" >>"$LOG" 2>&1 ;;
  scallop:sum)   cd "$ROOT/baselines/scallop/mnist_sum"
    "$PY" scallop_sum.py --n-epochs 1 --digits 2 --top-k 3 --seed 1234 >>"$LOG" 2>&1 ;;
  scallop:add)   cd "$ROOT/baselines/scallop/mnist_add"
    "$PY" scallop_add.py --n-epochs 1 --N 2 --k 3 --seed 1234 >>"$LOG" 2>&1 ;;
  deepproblog:add) cd "$ROOT/baselines/deepproblog/src/deepproblog/examples/MNIST"
    "$PY" -u addition.py >>"$LOG" 2>&1 ;;
  deepsoftlog:sum) cd "$ROOT/baselines/deepsoftlog"
    CFG="$ROOT/logs/smoke/_cfg_dsl_sum.yaml"
    sed -e 's/^nb_epochs:.*/nb_epochs: 1/' \
        deepsoftlog/experiments/mnist_sum/config4.yaml > "$CFG"
    PYTHONPATH=. "$PY" deepsoftlog/run_experiments.py mnist_sum "$CFG" >>"$LOG" 2>&1 ;;
  spl:sum)       cd "$ROOT/baselines/spl"
    # The real SPL path: pre-compiled SDD + semantic loss. Only mnistsum_N2 is
    # compiled; larger N needs compile_mnistsum_sdd.py first.
    "$PY" train_mnistsum_spl.py --N 2 --epochs 1 >>"$LOG" 2>&1 ;;
  *) echo "unknown combination $METHOD:$TASK" >&2; exit 2 ;;
esac
rc=$?
echo "# $METHOD $TASK  exit=$rc  ${SECONDS}s  $(date -Is)" >> "$LOG"
exit $rc
