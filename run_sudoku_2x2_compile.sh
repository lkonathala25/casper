#!/usr/bin/env bash
# Bounded 2x2 sudoku compile with live diagnostics tee'd to sudoku_2x2_compile.log.
set -u
cd "$(dirname "$0")"

LOG=sudoku_2x2_compile.log
OUT=examples/sudoku_2x2_compiled.txt
TIMEOUT=${TIMEOUT:-60s}

rm -f "$LOG" "$OUT"
timeout "$TIMEOUT" python3 src/resolution.py examples/sudoku_2x2.txt "$OUT" 2>&1 | tee "$LOG"
echo "exit=${PIPESTATUS[0]}"
