#!/usr/bin/env bash
# 4x4 sudoku compile with live diagnostics tee'd to sudoku_4x4_compile.log.
# Uses min-degree elimination ordering on the primal constraint graph; the
# input file's `ordering` line is overridden.
set -u
cd "$(dirname "$0")"

LOG=sudoku_4x4_compile.log
OUT=examples/sudoku_4x4_compiled.txt

rm -f "$LOG" "$OUT"
python3 src/resolution.py --ordering min-degree examples/sudoku_4x4.txt "$OUT" 2>&1 | tee "$LOG"
echo "exit=${PIPESTATUS[0]}"
