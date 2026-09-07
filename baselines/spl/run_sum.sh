#!/usr/bin/env bash
# SPL on MNIST-Sum: pre-compiled SDD + semantic loss.
# sdd_files/ ships mnistsum_N2 only; larger N needs compile_mnistsum_sdd.py
# first, which is where the exact method stops being tractable.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-$HOME/miniconda3/envs/casper_env/bin/python}"
NS="${NS:-2}"; EPOCHS="${EPOCHS:-30}"
LOGDIR="${LOGDIR:-../../logs/baselines}"; mkdir -p "$LOGDIR"
for n in $NS; do
  if [ ! -f "sdd_files/mnistsum_N${n}.sdd" ]; then
    echo ">>> compiling SDD for N=$n"; "$PY" compile_mnistsum_sdd.py --N "$n"
  fi
  echo ">>> SPL sum N=$n"
  "$PY" train_mnistsum_spl.py --N "$n" --epochs "$EPOCHS" \
    2>&1 | tee "$LOGDIR/spl_sum_N${n}.log"
done
