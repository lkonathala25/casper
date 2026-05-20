#!/usr/bin/env bash
# MNIST-Sum sweep: N in {2, 16, 64, 256, 1024} x 5 seeds x {closed_form, cvxpy}.
#
# Per the experimental setup: 5000 train / 1000 test samples for every N,
# batch size 16, Adam lr=1e-3, cosine schedule with eta_min=1e-5, 200 epochs.
# Mean +/- std is reported across the 5 seeds.
#
# CVXPY backend at large N does not converge inside per-iter / per-run budgets;
# train.py emits a NaN T/O row to the CSV when that happens. This is the
# "QP-based projection does not scale" failure mode.

set -e

# Run from the repo root regardless of invocation cwd.
cd "$(dirname "$0")"

OUTDIR=logs/mnist_sum
mkdir -p "$OUTDIR"

EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEEDS=(0 1 2 3 4)
N_VALUES=(2 16 64 256 1024)
BACKENDS=(closed_form cvxpy)

# Wall-clock guards — only relevant for the cvxpy backend at large N.
SOLVER_TIME_LIMIT="${SOLVER_TIME_LIMIT:-10}"
PER_ITER_TIMEOUT_S="${PER_ITER_TIMEOUT_S:-30}"
MAX_RUN_TIME_S="${MAX_RUN_TIME_S:-3600}"

echo "============================================="
echo "MNIST-Sum sweep"
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
        CSV="$OUTDIR/sum_n${N}_${BACKEND}.csv"
        rm -f "$CSV"
        for SEED in "${SEEDS[@]}"; do
            echo ""
            echo ">>> task=sum N=$N seed=$SEED backend=$BACKEND"
            set +e
            python src/train.py \
                --task sum \
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
