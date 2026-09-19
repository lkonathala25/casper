#!/usr/bin/env bash
# Start the MNIST CASPER sweep detached on one GPU, one process per seed running
# in parallel (each run needs at most ~1.6 GB and leaves the GPU mostly idle).
# Per-seed logs go to $RUNS_ROOT/mnist/casper/sweep_seed<s>_<time>.log; per-run
# results, checkpoints and logs go to each run folder. Finished runs (DONE) are
# skipped, so relaunching resumes where a sweep stopped. Extra env (TASKS,
# SUM_NS, ADD_NS, EPOCHS, ...) is passed through to run_casper.sh.
# Sweeps over different settings can run side by side (e.g. the main sweep on one
# GPU and a single large N on another): the pid file is per seed AND per setting,
# named from TASKS/SUM_NS/ADD_NS or from TAG when given.
# Usage: GPU=2 SEEDS="0 1 2" bash tasks/mnist/launch_casper.sh
#   stop one seed: kill -- -$(cat $RUNS_ROOT/mnist/casper/sweep_<tag>_seed0.pid)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
RUNS_ROOT="${RUNS_ROOT:-/Data/lk1325/runs}"
GPU="${GPU:-2}"
SEEDS="${SEEDS:-0 1 2}"
# One sweep per seed shares the box, so cap each process's CPU threads instead of
# letting every one of them claim all cores.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# train.log should follow the run, not appear in blocks when the pipe buffer fills.
export PYTHONUNBUFFERED=1
out="$RUNS_ROOT/mnist/casper"; mkdir -p "$out"
stamp="$(date +%Y%m%d_%H%M%S)"
tag="${TAG:-$(echo "${TASKS:-sumadd}_${SUM_NS:-all}_${ADD_NS:-all}${ADD_PROTOCOL:+_$ADD_PROTOCOL}" | tr -s ' ' '-' | tr -cd 'A-Za-z0-9_-')}"
for s in $SEEDS; do
  pidfile="$out/sweep_${tag}_seed$s.pid"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "Seed $s of this sweep ($tag) is already running (pid $(cat "$pidfile")); skipping." >&2
    continue
  fi
  log="$out/sweep_${tag}_seed${s}_${stamp}.log"
  CUDA_VISIBLE_DEVICES="$GPU" RUNS_ROOT="$RUNS_ROOT" SEEDS="$s" \
    setsid nohup bash tasks/mnist/run_casper.sh > "$log" 2>&1 < /dev/null &
  echo $! > "$pidfile"
  echo "seed $s -> pid $!  log $log"
done
echo "Runs: $out/{sum,add}/n<N>/seed<s>/"
