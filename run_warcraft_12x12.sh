#!/usr/bin/env bash
# Warcraft 12x12 sweep: \layer{} (full method) over 5 seeds.
#
# Per the experimental setup: ResNet-18 trained from scratch, Adam lr=5e-4 with
# cosine schedule (eta_min=5e-6 = lr*0.01), batch size 128, 300 epochs with
# early stopping (patience 30), surrogate threshold eps=1e-3, auxiliary L2
# self-distillation loss between unprojected and projected per-node
# distributions weighted by lambda=1.0.
#
# Variables are processed in row-major order. Set PROCESSING_ORDER=diagonal
# for the alternative ordering (empirically equivalent on this task).
#
# Outputs per-seed CSVs and checkpoints to logs/warcraft_12x12/.

set -e

# Run from the repo root regardless of invocation cwd.
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-data/warcraft_shortest_path_oneskin/12x12/preprocessed_4conn}"
OUTDIR="${OUTDIR:-logs/warcraft_12x12}"
mkdir -p "$OUTDIR"

GRID_SIZE=12
PROCESSING_ORDER="${PROCESSING_ORDER:-row}"      # row | diagonal

# Optimisation
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-5e-4}"
PATIENCE="${PATIENCE:-30}"
SCHEDULER="${SCHEDULER:-cosine}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# CRL / MinEdit + L2 self-distillation
LAMBDA_NEG="${LAMBDA_NEG:-1.0}"
LAMBDA_L2_EDIT="${LAMBDA_L2_EDIT:-1.0}"          # paper lambda = 1.0

# Auxiliary losses retained from the original 12x12 recipe
BETA_CE="${BETA_CE:-0.1}"
GAMMA_FOCAL="${GAMMA_FOCAL:-2.0}"
LAMBDA_EDGE="${LAMBDA_EDGE:-0.5}"
LAMBDA_PAIR="${LAMBDA_PAIR:-0.3}"
LAMBDA_LEN="${LAMBDA_LEN:-0.05}"

SEEDS=(0 1 2 3 4)

echo "============================================="
echo "Warcraft 12x12 sweep — \\layer{} (full method)"
echo "  Output:            $OUTDIR/"
echo "  Processing order:  $PROCESSING_ORDER"
echo "  Epochs / patience: $EPOCHS / $PATIENCE"
echo "  Batch size:        $BATCH_SIZE"
echo "  LR / scheduler:    $LR / $SCHEDULER"
echo "  lambda_l2_edit:    $LAMBDA_L2_EDIT"
echo "  lambda_neg:        $LAMBDA_NEG"
echo "  beta_ce / focal:   $BETA_CE / $GAMMA_FOCAL"
echo "  edge / pair / len: $LAMBDA_EDGE / $LAMBDA_PAIR / $LAMBDA_LEN"
echo "  Seeds:             ${SEEDS[*]}"
echo "  Started:           $(date)"
echo "============================================="

for SEED in "${SEEDS[@]}"; do
    SEED_DIR="$OUTDIR/seed${SEED}"
    mkdir -p "$SEED_DIR"

    echo ""
    echo ">>> seed=$SEED"
    set +e
    python src/train.py \
        --data_dir "$DATA_DIR" \
        --save_dir "$SEED_DIR" \
        --grid-size "$GRID_SIZE" \
        --condition crl_in_training \
        --n-runs 1 \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --lr "$LR" \
        --patience "$PATIENCE" \
        --scheduler "$SCHEDULER" \
        --grad-clip "$GRAD_CLIP" \
        --seed "$SEED" \
        --processing-order "$PROCESSING_ORDER" \
        --lambda-neg "$LAMBDA_NEG" \
        --beta-ce "$BETA_CE" \
        --gamma-focal "$GAMMA_FOCAL" \
        --lambda-edge "$LAMBDA_EDGE" \
        --lambda-pair "$LAMBDA_PAIR" \
        --lambda-len "$LAMBDA_LEN" \
        --lambda-l2-edit "$LAMBDA_L2_EDIT" \
        2>&1 | tee "$SEED_DIR/run.log"
    set -e
done

echo ""
echo "============================================="
echo "Finished: $(date)"
echo "Per-seed checkpoints + CSVs in $OUTDIR/seed{0..4}/"
echo "Run cost-ratio + Hamming + Consistent aggregation with:"
echo "  python src/cost_ratio.py --ckpt_dir <one-seed-dir>/crl_in_training \\"
echo "    --data_dir $DATA_DIR --grid-size $GRID_SIZE \\"
echo "    --processing-order $PROCESSING_ORDER --out_json $OUTDIR/cost_ratio.json"
echo "============================================="
