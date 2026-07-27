#!/usr/bin/env bash
# Compilation time vs. constraint-graph topology: banded (`exp`) against
# spanning-tree (`poly`), swept over m and bandwidth b. Writes
# compile_topology_timing.{csv,md,png} and tees the log.
set -u
cd "$(dirname "$0")"

LOG=compile_topology_timing.log
rm -f "$LOG"

python3 examples/bench_topology_compile.py \
    --m 20 50 100 200 \
    --graphs exp poly \
    --bands 1 2 3 4 5 \
    --classes 6 \
    --tau 0.5 \
    --clauses-per-edge 2 \
    --trials 3 \
    --timeout-s 120 \
    --prefix compile_topology_timing 2>&1 | tee "$LOG"
echo "exit=${PIPESTATUS[0]}"
