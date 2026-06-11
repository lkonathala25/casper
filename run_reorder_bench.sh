#!/usr/bin/env bash
# A/B benchmark for the group-reordering tautology-pruning optimization.
#
# For each variable, re-runs its elimination step IN ISOLATION (from the
# compiled artifact) twice — reordering OFF (baseline) then ON (new) — saving
# full verbose logs and printing a one-line OFF-vs-ON comparison.
#
# What to look for:
#   kept          -> MUST match OFF vs ON (proves the output is unchanged)
#   expand_s      -> the speedup (lower with reorder ON)
#   prefix_taut   -> tautologies pruned early (higher with reorder ON = mechanism working)
#
# Usage:
#   ./run_reorder_bench.sh                 # default: y_2 y_3
#   ./run_reorder_bench.sh y_2 y_3 y_5     # pick variables
#   ON_ONLY=1 ./run_reorder_bench.sh y_4   # skip the (18h) OFF baseline
#   TIMEOUT=900 ./run_reorder_bench.sh y_3 # cap each run at 900s
set -u
cd "$(dirname "$0")"

COMPILED=${COMPILED:-examples/sudoku_4x4_compiled.txt}
LOGDIR=${LOGDIR:-logs}
ON_ONLY=${ON_ONLY:-0}
TIMEOUT=${TIMEOUT:-0}            # seconds; 0 = no limit
VARS=("$@"); [ ${#VARS[@]} -eq 0 ] && VARS=(y_2 y_3)

mkdir -p "$LOGDIR"
run() {  # $1=flag(0/1) $2=var $3=logfile
  local cmd=(python3 src/bench_step.py "$COMPILED" "$2")
  [ "$TIMEOUT" -gt 0 ] && cmd=(timeout "$TIMEOUT" "${cmd[@]}")
  echo ">>> var=$2 SAT_REORDER_GROUPS=$1 -> $3"
  SAT_REORDER_GROUPS="$1" "${cmd[@]}" 2>&1 | tee "$3"
}
field() { grep -oE "$2=[0-9.]+" "$1" 2>/dev/null | tail -1 | cut -d= -f2; }

printf '\n%-6s %-4s %12s %12s %14s %10s\n' VAR RUN kept raw prefix_taut expand_s
echo "---------------------------------------------------------------------"
for v in "${VARS[@]}"; do
  off_log="$LOGDIR/${v}_off.log"; on_log="$LOGDIR/${v}_on.log"
  if [ "$ON_ONLY" != "1" ]; then
    run 0 "$v" "$off_log" >/dev/null
    printf '%-6s %-4s %12s %12s %14s %10s\n' "$v" OFF \
      "$(field "$off_log" kept)" "$(field "$off_log" raw)" \
      "$(field "$off_log" prefix_taut)" "$(field "$off_log" expand_s)"
  fi
  run 1 "$v" "$on_log" >/dev/null
  printf '%-6s %-4s %12s %12s %14s %10s\n' "$v" ON \
    "$(field "$on_log" kept)" "$(field "$on_log" raw)" \
    "$(field "$on_log" prefix_taut)" "$(field "$on_log" expand_s)"
  if [ "$ON_ONLY" != "1" ]; then
    ko=$(field "$off_log" kept); kn=$(field "$on_log" kept)
    if [ "$ko" = "$kn" ] && [ -n "$ko" ]; then
      echo "  [correctness OK] kept identical ($ko)"
    else
      echo "  [CHECK] kept differs OFF=$ko ON=$kn (or run incomplete)"
    fi
  fi
done
echo "---------------------------------------------------------------------"
echo "Full logs in $LOGDIR/  (per-pattern + heartbeat detail for the write-up)"
