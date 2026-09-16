#!/usr/bin/env bash
# Start the MNIST CASPER sweep (tasks/mnist/run_casper.sh) detached on one GPU, so
# it survives logout. The sweep log goes to $RUNS_ROOT/mnist/casper/sweep_<time>.log;
# per-run results, checkpoints and logs go to each run folder. Refuses to start a
# second sweep while one is running. Finished runs (DONE) are skipped, so
# relaunching resumes where a sweep stopped. Extra env (TASKS, SEEDS, ...) is
# passed through to run_casper.sh.
# Usage: GPU=2 bash tasks/mnist/launch_casper.sh
#   stop: kill -- -$(cat $RUNS_ROOT/mnist/casper/sweep.pid)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
RUNS_ROOT="${RUNS_ROOT:-/Data/lk1325/runs}"
GPU="${GPU:-2}"
out="$RUNS_ROOT/mnist/casper"; mkdir -p "$out"
pidfile="$out/sweep.pid"
if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
  echo "A sweep is already running (pid $(cat "$pidfile")); not starting another." >&2
  exit 1
fi
log="$out/sweep_$(date +%Y%m%d_%H%M%S).log"
CUDA_VISIBLE_DEVICES="$GPU" RUNS_ROOT="$RUNS_ROOT" setsid nohup bash tasks/mnist/run_casper.sh > "$log" 2>&1 < /dev/null &
echo $! > "$pidfile"
echo "Started MNIST CASPER sweep on GPU $GPU (pid $!)"
echo "  sweep log: $log"
echo "  runs:      $out/{sum,add}/n<N>/seed<s>/"
