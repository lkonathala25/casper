#!/usr/bin/env bash
# Start the MNIST CASPER sweep detached on one GPU, one process per seed running
# in parallel (each run needs at most ~1.6 GB and leaves the GPU mostly idle).
# Per-seed logs go to $RUNS_ROOT/mnist/casper/sweep_seed<s>_<time>.log; per-run
# results, checkpoints and logs go to each run folder. Finished runs (DONE) are
# skipped, so relaunching resumes where a sweep stopped. Extra env (TASKS,
# SUM_NS, ADD_NS, EPOCHS, ...) is passed through to run_casper.sh.
# Usage: GPU=2 SEEDS="0 1 2" bash tasks/mnist/launch_casper.sh
#   stop one seed: kill -- -$(cat $RUNS_ROOT/mnist/casper/sweep_seed0.pid)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
RUNS_ROOT="${RUNS_ROOT:-/Data/lk1325/runs}"
GPU="${GPU:-2}"
SEEDS="${SEEDS:-0 1 2}"
# One sweep per seed shares the box, so cap each process's CPU threads instead of
# letting every one of them claim all cores.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
out="$RUNS_ROOT/mnist/casper"; mkdir -p "$out"
stamp="$(date +%Y%m%d_%H%M%S)"
for s in $SEEDS; do
  pidfile="$out/sweep_seed$s.pid"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "Seed $s is already running (pid $(cat "$pidfile")); skipping." >&2
    continue
  fi
  log="$out/sweep_seed${s}_${stamp}.log"
  CUDA_VISIBLE_DEVICES="$GPU" RUNS_ROOT="$RUNS_ROOT" SEEDS="$s" \
    setsid nohup bash tasks/mnist/run_casper.sh > "$log" 2>&1 < /dev/null &
  echo $! > "$pidfile"
  echo "seed $s -> pid $!  log $log"
done
echo "Runs: $out/{sum,add}/n<N>/seed<s>/"
