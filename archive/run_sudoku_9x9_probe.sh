#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"

LOG=sudoku_9x9_probe.log
OUT=examples/sudoku_9x9_compiled.txt

rm -f "$LOG" "$OUT"
echo "# probe start: $(date -Is)" | tee -a "$LOG"
python3 src/resolution.py --ordering min-degree examples/sudoku_9x9.txt "$OUT" 2>&1 | tee -a "$LOG"
echo "exit=${PIPESTATUS[0]}  # end: $(date -Is)" | tee -a "$LOG"
