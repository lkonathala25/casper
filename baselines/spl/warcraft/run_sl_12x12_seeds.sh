#!/bin/bash
# Run Semantic Loss (BCE + lambda * -log WMC) on Warcraft 12x12 across 3 seeds.
# Trainer: SL  (NOT FFCircuit/Circuit -- those are SPL).
#
# Usage: ./run_sl_12x12_seeds.sh
# Run from inside WarcraftShortestPath/.

set -euo pipefail

cd "$(dirname "$0")"

BASE_CFG="settings/warcraft_shortest_path/sl/12x12.json"
SEEDS=(1 2 3)
SL_WEIGHT=0.01

mkdir -p checkpoints logs

for SEED in "${SEEDS[@]}"; do
    MODEL_DIR="results/warcraft_sl_12x12/seed${SEED}"
    LOG_FILE="logs/sl_12x12_seed${SEED}.log"
    CKPT_NAME="checkpoints/sl_12x12_seed${SEED}_w${SL_WEIGHT}.pt"

    echo "=========================================================="
    echo "  Seed=${SEED}  sl_weight=${SL_WEIGHT}  -> ${MODEL_DIR}"
    echo "=========================================================="

    PARAMS=$(python -c "
import json, sys
print(json.dumps({
    'default_json': '${BASE_CFG}',
    'seed': ${SEED},
    'sl_weight': ${SL_WEIGHT},
    'model_dir': '${MODEL_DIR}',
}))
")

    python main.py "${PARAMS}" 2>&1 | tee "${LOG_FILE}"

    if [[ -f "checkpoints/sl_${SL_WEIGHT}.pt" ]]; then
        cp "checkpoints/sl_${SL_WEIGHT}.pt" "${CKPT_NAME}"
        echo "Saved checkpoint -> ${CKPT_NAME}"
    fi
done

echo "All seeds done. Per-seed results in results/warcraft_sl_12x12/seed{1,2,3}/"
