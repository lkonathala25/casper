#!/usr/bin/env bash
# Capped 9x9 compile: derives only minimal conflicts up to CAP arity, optionally
# skipping any single pattern whose expansion exceeds MAX_EXPANSION. Tractable
# but under-constrained (no satisfaction guarantee). Launch with:
#   CAP=5 ./run_sudoku_9x9_cap.sh                          # arity cap only
#   CAP=6 MAX_EXPANSION=2000000 ./run_sudoku_9x9_cap.sh    # cap + cost guard
#   CAP=2 SAT_WORKERS=128 ./run_sudoku_9x9_cap.sh
set -u
cd "$(dirname "$0")"

CAP="${CAP:-5}"
WORKERS="${SAT_WORKERS:-128}"
LOG="sudoku_9x9_cap${CAP}.log"
OUT="examples/sudoku_9x9_cap${CAP}_compiled.txt"

EXP_ARGS=()
if [[ -n "${MAX_EXPANSION:-}" ]]; then
    EXP_ARGS+=(--max-expansion "$MAX_EXPANSION")
fi

rm -f "$LOG" "$OUT"
echo "# cap=$CAP workers=$WORKERS max_expansion=${MAX_EXPANSION:-unlimited} cores=$(nproc) start: $(date -Is)" | tee -a "$LOG"
SAT_WORKERS="$WORKERS" python3 src/resolution.py \
    --ordering min-degree --cap "$CAP" "${EXP_ARGS[@]}" \
    examples/sudoku_9x9.txt "$OUT" 2>&1 | tee -a "$LOG"
echo "exit=${PIPESTATUS[0]}  # end: $(date -Is)" | tee -a "$LOG"
