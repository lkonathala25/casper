# Shared helpers for the experiment run scripts. Source after setting ROOT.
# One folder per run under $RUNS_ROOT with train.log inside; finished runs are
# marked DONE and skipped on relaunch (FORCE=1 reruns); DRY_RUN=1 only prints.
RUNS_ROOT="${RUNS_ROOT:-/Data/lk1325/runs}"
PY="${PY:-$HOME/miniconda3/envs/casper_env/bin/python}"
SEEDS="${SEEDS:-0 1 2 3 4}"
RUN_FAILURES=0

run_one() {
  local dir=$1; shift
  if [[ "${FORCE:-0}" != 1 && -f "$dir/DONE" ]]; then
    echo "[$(date +%T)] skip   $dir (DONE)"; return 0
  fi
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "[dry-run] $dir"; printf '   '; printf ' %q' "$@"; echo; return 0
  fi
  mkdir -p "$dir"
  { echo "# $(date '+%F %T') host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} commit=$(git -C "$ROOT" describe --always --dirty)"
    printf '#'; printf ' %q' "$@"; echo; } > "$dir/train.log"
  echo "[$(date +%T)] start  $dir"
  if "$@" >> "$dir/train.log" 2>&1; then
    touch "$dir/DONE"; echo "[$(date +%T)] done   $dir"
  else
    local rc=$?
    RUN_FAILURES=$((RUN_FAILURES + 1))
    echo "[$(date +%T)] FAILED $dir (exit $rc, see train.log)"
  fi
}

finish_runs() {
  if (( RUN_FAILURES > 0 )); then echo "$RUN_FAILURES run(s) failed"; return 1; fi
}
