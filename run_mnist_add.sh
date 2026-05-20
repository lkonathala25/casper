#!/usr/bin/env bash
# MNIST-Add sweep: N in {2, 15, 100, 256} x 5 seeds x {closed_form, cvxpy}.
#
# Per the experimental setup: floor(60000/2N) train, floor(10000/2N) test
# samples, batch size 64, Adam lr=1e-3 (3e-4 for N in {100, 256}), cosine
# schedule with eta_min=1e-5, 100 epochs. Mean +/- std across 5 seeds.
#
# Carry-chain decomposition is used for every N — there is no flat sum head
# fallback. train.py wires in the lr override automatically based on N.
#
# CVXPY backend at large N does not converge inside per-iter / per-run budgets;
# train.py emits a NaN T/O row to the CSV when that happens.

set -e

# Run from the repo root regardless of invocation cwd.
cd "$(dirname "$0")"

OUTDIR=logs/mnist_add
mkdir -p "$OUTDIR"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEEDS=(0 1 2 3 4)
N_VALUES=(2 15 100 256)
BACKENDS=(closed_form cvxpy)

# Wall-clock guards — only relevant for the cvxpy backend at large N.
SOLVER_TIME_LIMIT="${SOLVER_TIME_LIMIT:-10}"
PER_ITER_TIMEOUT_S="${PER_ITER_TIMEOUT_S:-30}"
MAX_RUN_TIME_S="${MAX_RUN_TIME_S:-3600}"

echo "============================================="
echo "MNIST-Add sweep (carry-chain)"
echo "  Output:     $OUTDIR/"
echo "  Epochs:     $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Seeds:      ${SEEDS[*]}"
echo "  N values:   ${N_VALUES[*]}"
echo "  Backends:   ${BACKENDS[*]}"
echo "  Started:    $(date)"
echo "============================================="

for BACKEND in "${BACKENDS[@]}"; do
    for N in "${N_VALUES[@]}"; do
        CSV="$OUTDIR/add_n${N}_${BACKEND}.csv"
        rm -f "$CSV"
        for SEED in "${SEEDS[@]}"; do
            echo ""
            echo ">>> task=add N=$N seed=$SEED backend=$BACKEND"
            set +e
            python src/train.py \
                --task add \
                --n-digits "$N" \
                --seed "$SEED" \
                --epochs "$EPOCHS" \
                --batch-size "$BATCH_SIZE" \
                --projection-backend "$BACKEND" \
                --solver-time-limit "$SOLVER_TIME_LIMIT" \
                --per-iter-timeout-s "$PER_ITER_TIMEOUT_S" \
                --max-run-time-s "$MAX_RUN_TIME_S" \
                --csv "$CSV"
            set -e
        done
    done
done

echo ""
echo "============================================="
echo "Finished: $(date)"
echo "Per-(N, backend) CSVs in $OUTDIR/"
echo "============================================="
