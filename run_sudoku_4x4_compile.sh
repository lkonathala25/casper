#!/usr/bin/env bash
# Bounded 4x4 sudoku compile with live diagnostics tee'd to sudoku_4x4_compile.log.
set -u
cd "$(dirname "$0")"

LOG=sudoku_4x4_compile.log
OUT=examples/sudoku_4x4_compiled.txt
TIMEOUT=${TIMEOUT:-1800s}

rm -f "$LOG" "$OUT"
timeout "$TIMEOUT" python3 src/resolution.py examples/sudoku_4x4.txt "$OUT" 2>&1 | tee "$LOG"
echo "exit=${PIPESTATUS[0]}"
