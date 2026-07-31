#!/usr/bin/env bash
# Same sweep as the topology experiment, with elimination ordering as the only
# varied axis: structural (min-fill), high/low confidence, and random.
# No per-compile cap, so every reported time is a true measurement; the sweep
# runs m-major and appends each finished cell to the CSV, so an interrupted run
# still leaves complete results for every m it reached.
set -u
cd "$(dirname "$0")"

LOG=compile_ordering_confidence.log
rm -f "$LOG"

python3 -u examples/bench_ordering_confidence.py \
    --m 20 50 100 200 \
    --graphs exp poly \
    --bands 1 2 3 4 5 \
    --classes 6 \
    --tau 0.5 \
    --clauses-per-edge 2 \
    --radius 10.0 \
    --lam 0.5 \
    --eps 1.0 \
    --trials 3 \
    --heavy-trials 3 \
    --heavy-from-band 3 \
    --per-compile-timeout 0 \
    --prefix compile_ordering_confidence 2>&1 | tee "$LOG"
echo "exit=${PIPESTATUS[0]}"
