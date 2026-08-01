#!/usr/bin/env bash
# Warcraft 12x12 \layer{} ablation: training-only / inference-only / training-no-L2-no-aux.
#
# All three conditions strip every auxiliary loss (lambda_edge / lambda_pair /
# lambda_len / beta_ce / gamma_focal = 0) and the L2 self-distillation term
# (lambda_l2_edit = 0), so each row is a pure ablation of where the MinEdit
# projection is applied:
#
#   training-only           : MinEdit at training only; CRL DISABLED at inference.
#                             (--condition crl_in_training_eval_off)
#   inference-only          : plain weighted CE at training; CRL applied at inference.
#                             (--condition crl_inference_only)
#   training-no-L2-no-aux   : MinEdit at both training and inference, no aux, no L2.
#                             (--condition crl_in_training)
#
# 5 seeds (0..4) per condition. All other hyperparameters identical to the
# main run_warcraft_12x12.sh: ResNet-18 from scratch, Adam lr=5e-4, cosine
# schedule, batch 128, 300 epochs, patience 30, eps=1e-3.
#
# Outputs per-(condition, seed) checkpoints + CSVs to logs/warcraft_ablation_12x12/.

set -e

# Run from the repo root regardless of invocation cwd.
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-data/warcraft_shortest_path_oneskin/12x12/preprocessed_4conn}"
OUTDIR="${OUTDIR:-logs/warcraft_ablation_12x12}"
mkdir -p "$OUTDIR"

GRID_SIZE=12
PROCESSING_ORDER="${PROCESSING_ORDER:-row}"

EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-5e-4}"
PATIENCE="${PATIENCE:-30}"
SCHEDULER="${SCHEDULER:-cosine}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

LAMBDA_NEG="${LAMBDA_NEG:-1.0}"

# All zero — pure MinEdit ablation, no aux, no L2.
LAMBDA_L2_EDIT=0.0
BETA_CE=0.0
GAMMA_FOCAL=0.0
LAMBDA_EDGE=0.0
LAMBDA_PAIR=0.0
LAMBDA_LEN=0.0
SKIP_WEIGHT=1.0
PATH_WEIGHT=1.0

CONDITIONS=(crl_in_training_eval_off crl_inference_only crl_in_training)
SEEDS=(0 1 2 3 4)

echo "============================================="
echo "Warcraft 12x12 ablation — pure MinEdit (no aux, no L2)"
echo "  Output:            $OUTDIR/"
echo "  Processing order:  $PROCESSING_ORDER"
echo "  Epochs / patience: $EPOCHS / $PATIENCE"
echo "  Conditions:        ${CONDITIONS[*]}"
echo "  Seeds:             ${SEEDS[*]}"
echo "  Started:           $(date)"
echo "============================================="

for COND in "${CONDITIONS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        SEED_DIR="$OUTDIR/${COND}/seed${SEED}"
        mkdir -p "$SEED_DIR"

        echo ""
        echo ">>> condition=$COND seed=$SEED"
        set +e
        python src/train.py \
            --data_dir "$DATA_DIR" \
            --save_dir "$SEED_DIR" \
            --grid-size "$GRID_SIZE" \
            --condition "$COND" \
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
            --skip-weight "$SKIP_WEIGHT" \
            --path-weight "$PATH_WEIGHT" \
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
done

echo ""
echo "============================================="
echo "Finished: $(date)"
echo "Per-(condition, seed) results in $OUTDIR/<condition>/seed{0..4}/"
echo "Aggregate per condition with:"
echo "  python src/compute_metrics.py --ckpt-dir $OUTDIR/<condition> \\"
echo "    --data_dir $DATA_DIR --grid-size $GRID_SIZE \\"
echo "    --processing-order $PROCESSING_ORDER \\"
echo "    --out_json $OUTDIR/<condition>/metrics.json"
echo "============================================="
