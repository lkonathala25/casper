# Shared helpers for the experiment run scripts. Source after setting ROOT.
# One folder per run under $RUNS_ROOT with train.log inside; finished runs are
# marked DONE and skipped on relaunch (FORCE=1 reruns); DRY_RUN=1 only prints.
# train.log is appended to, so a restarted run keeps the log of earlier attempts.
# A run holds an flock on <dir>/.lock while it trains, so a second launch into
# the same folder is skipped instead of training alongside it.
RUNS_ROOT="${RUNS_ROOT:-$ROOT/runs}"
PY="${PY:-python}"
SEEDS="${SEEDS:-0 1 2 3 4}"
RUN_FAILURES=0

run_commit() {  # commit of $1, reading .git directly when git is not on PATH
  git -C "$1" describe --always --dirty 2>/dev/null && return
  local head; head=$(cat "$1/.git/HEAD" 2>/dev/null) || return
  case "$head" in
    "ref: "*) cut -c1-12 "$1/.git/${head#ref: }" 2>/dev/null ;;
    *) printf '%.12s\n' "$head" ;;
  esac
}

run_one() {
  local dir=$1; shift
  if [[ "${FORCE:-0}" != 1 && -f "$dir/DONE" ]]; then
    echo "[$(date +%T)] skip   $dir (DONE)"; return 0
  fi
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "[dry-run] $dir"; printf '   '; printf ' %q' "$@"; echo; return 0
  fi
  mkdir -p "$dir"
  local lock_fd
  exec {lock_fd}>"$dir/.lock"
  if ! flock -n "$lock_fd"; then
    exec {lock_fd}>&-
    echo "[$(date +%T)] busy   $dir (another process is training it)"; return 0
  fi
  if [[ "${FORCE:-0}" != 1 && -f "$dir/DONE" ]]; then
    exec {lock_fd}>&-
    echo "[$(date +%T)] skip   $dir (DONE)"; return 0
  fi
  { echo "# $(date '+%F %T') host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} commit=$(run_commit "$ROOT")"
    printf '#'; printf ' %q' "$@"; echo; } >> "$dir/train.log"
  echo "[$(date +%T)] start  $dir"
  if "$@" >> "$dir/train.log" 2>&1; then  # the child inherits the lock fd, so the lock lives as long as it does
    touch "$dir/DONE"; echo "[$(date +%T)] done   $dir"
  else
    local rc=$?
    RUN_FAILURES=$((RUN_FAILURES + 1))
    echo "[$(date +%T)] FAILED $dir (exit $rc, see train.log)"
  fi
  exec {lock_fd}>&-
}

casper_variant() {  # MODE L2_PENALTY [BACKEND]: run-folder name of a CASPER configuration
  local v=$1
  case "$1" in
    train_infer|train_only)
      [[ -n "$2" ]] || { echo "casper_variant: MODE=$1 needs L2_PENALTY (1 = full CASPER, 0 = w/o l2 penalty)" >&2; return 1; }
      v+="_l2_penalty$2" ;;
    infer_only|none) [[ -z "$2" ]] || { echo "casper_variant: L2_PENALTY needs the layer in training, not MODE=$1" >&2; return 1; } ;;
    *) echo "casper_variant: unknown MODE=$1" >&2; return 1 ;;
  esac
  [[ "${3:-closed_form}" != closed_form ]] && v+="_$3"
  echo "$v"
}

finish_runs() {
  if (( RUN_FAILURES > 0 )); then echo "$RUN_FAILURES run(s) failed"; return 1; fi
}
