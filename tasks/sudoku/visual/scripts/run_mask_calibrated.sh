#!/usr/bin/env bash
# Transformer neuro-solver, CASPER edit layer: mask-only calibration (mask_nn
# trained; perception + nn_solver frozen) for each seed, then -test evaluation
# of all seeds. Reference recipe = llm-sudoku seed-0 run (spec §2.3, §4.5).
#
# Detached launch (from tasks/sudoku/visual):
#   mkdir -p results/mask_calibrated
#   SEEDS="0 1 2 3 4 5 6 7 8 9" GPU_ID=0 \
#     setsid nohup scripts/run_mask_calibrated.sh > results/mask_calibrated/driver.log 2>&1 < /dev/null &
#
# Env:
#   SEEDS       space-separated seeds, run sequentially (default "0")
#   GPU_ID      GPU to use (default 0)
#   PYTHON_BIN  python of the `sudoku` env
#               (default <conda envs dir>/sudoku/bin/python, e.g.
#               /home/la2324/miniconda3/envs/sudoku/bin/python)
#   EXTRA_ARGS  extra flags appended to minedit_mask_calibrate.py
#               (e.g. "--epochs 1" for the smoke gate P4)
#   CASPER_GROUP_SOLVER  greedy (default) or search; passed to training and
#               evaluation as --casper-group-solver. search runs get their own
#               run name (suffix _search) and log dir (mask_calibrated_search).
# Locations (checkpoints, logs) come from paths.py; override with its SUDOKU_* env vars.
set -euo pipefail

VISUAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$VISUAL_DIR"
eval "$(python3 paths.py --shell)"

SEEDS="${SEEDS:-0}"
GPU="${GPU_ID:-0}"
PY="${PYTHON_BIN:-$(dirname "$SWIPL_ENV")/sudoku/bin/python}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
GROUP_SOLVER="${CASPER_GROUP_SOLVER:-greedy}"
case "$GROUP_SOLVER" in
    greedy) SUFFIX="" ;;
    search) SUFFIX="_search" ;;
    *) echo "CASPER_GROUP_SOLVER must be greedy or search, got '$GROUP_SOLVER'" >&2; exit 2 ;;
esac
RUN_NAME="e10_lr1e-4_wd3e-1_bs200_cos_wu0_nw20_clip1_it3-10_pat5_d1_bs0_-1_-2_casper${SUFFIX}"
MODEL_DIR="$CKPT_DIR/casper/mask_calibrated/$RUN_NAME"
LOG_DIR="$RESULTS_DIR/mask_calibrated${SUFFIX}"
SEEDS_CSV="$(echo $SEEDS | tr ' ' ',')"

mkdir -p "$MODEL_DIR" "$LOG_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(stamp)] seeds='${SEEDS}' gpu=${GPU} python=${PY} group_solver=${GROUP_SOLVER} extra='${EXTRA_ARGS}'"
echo "[$(stamp)] model dir ${MODEL_DIR}; logs ${LOG_DIR}"

for S in $SEEDS; do
    echo "[$(stamp)] training seed ${S}"
    "$PY" -u minedit_mask_calibrate.py \
        --data satnet \
        --seed "$S" \
        --epochs 10 \
        --batch-size 200 \
        --lr 1e-4 \
        --weight-decay 3e-1 \
        --warmup 0 \
        --neg-weight 20 \
        --clip-norm 1.0 \
        --minedit-iters-train 3 \
        --minedit-iters-eval 10 \
        --patience 5 \
        --min-delta 1 \
        --eval-mask-biases "0.0,-1.0,-2.0" \
        --edit-layer casper \
        --casper-group-solver "$GROUP_SOLVER" \
        --model-dir "$MODEL_DIR" \
        ${EXTRA_ARGS} \
        2>&1 | tee "$LOG_DIR/train_seed${S}.log"
done

echo "[$(stamp)] evaluating seeds ${SEEDS_CSV} on -test"
"$PY" -u evaluate_checkpoints.py \
    --folder "$MODEL_DIR" \
    --seeds "$SEEDS_CSV" \
    --minedit \
    --edit-layer casper \
    --casper-group-solver "$GROUP_SOLVER" \
    --python-bin "$PY" \
    2>&1 | tee "$LOG_DIR/eval_seeds_${SEEDS_CSV//,/_}.log"

echo "[$(stamp)] DONE. Summary: ${MODEL_DIR}/evaluation_summary.log"
