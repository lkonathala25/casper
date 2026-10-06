#!/usr/bin/env bash
# LLM backbone (Qwen2.5-14B-Instruct 4-bit, SATNet LoRA SFT), CASPER edit layer,
# fusion head FROZEN: mask-only calibration for each seed, then -test evaluation
# of all seeds. Recipe = llm-sudoku train_llm_5seeds.sh + --edit-layer casper
# --freeze-fusion-head (spec §2.3, §4.5). LLM boards come from the read-only
# cache in paths.LLM_CACHE_DIR; no generation happens unless FILL_CACHE=1.
#
# Detached launch (from tasks/sudoku/visual):
#   mkdir -p results/llm_mask_calibrated
#   SEEDS="0 1 2 3 4" GPU_ID=0 \
#     setsid nohup scripts/run_llm_mask_calibrated.sh > results/llm_mask_calibrated/driver.log 2>&1 < /dev/null &
#
# Env:
#   SEEDS       space-separated seeds, run sequentially (default "0")
#   GPU_ID      GPU to use (default 0)
#   PYTHON_BIN  python of the `sudoku-fine` env
#               (default <conda envs dir>/sudoku-fine/bin/python, e.g.
#               /home/la2324/miniconda3/envs/sudoku-fine/bin/python)
#   EXTRA_ARGS  extra flags appended to minedit_llm_mask_calibrate.py
#               (e.g. "--epochs 1" for the smoke gate P5)
#   FILL_CACHE  1 = run fill_llm_cache.py first (generates missing boards and
#               WRITES the cache); default 0 = cache is read-only
#   CASPER_GROUP_SOLVER  greedy (default) or search; passed to training and
#               evaluation as --casper-group-solver. search runs get their own
#               checkpoint dir (llm_mask_calibrated_search) and log dir.
# Before training, a cache-coverage check runs on the GPU (perception only, no
# LLM) and aborts if test < 483/500 or train < 8847/9000 boards are cached.
set -euo pipefail

VISUAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$VISUAL_DIR"
eval "$(python3 paths.py --shell)"

SEEDS="${SEEDS:-0}"
GPU="${GPU_ID:-0}"
PY="${PYTHON_BIN:-$(dirname "$SWIPL_ENV")/sudoku-fine/bin/python}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
GROUP_SOLVER="${CASPER_GROUP_SOLVER:-greedy}"
case "$GROUP_SOLVER" in
    greedy) SUFFIX="" ;;
    search) SUFFIX="_search" ;;
    *) echo "CASPER_GROUP_SOLVER must be greedy or search, got '$GROUP_SOLVER'" >&2; exit 2 ;;
esac
CKPT_BASE="$CKPT_DIR/casper/llm_mask_calibrated${SUFFIX}"
LOG_DIR="$RESULTS_DIR/llm_mask_calibrated${SUFFIX}"
SEEDS_CSV="$(echo $SEEDS | tr ' ' ',')"

mkdir -p "$CKPT_BASE" "$LOG_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(stamp)] seeds='${SEEDS}' gpu=${GPU} python=${PY} group_solver=${GROUP_SOLVER} extra='${EXTRA_ARGS}'"
echo "[$(stamp)] checkpoints ${CKPT_BASE}; logs ${LOG_DIR}; LLM cache ${LLM_CACHE_DIR}"

# ---- LLM cache ---------------------------------------------------------------
if [[ "${FILL_CACHE:-0}" == "1" ]]; then
    echo "[$(stamp)] FILL_CACHE=1: generating boards missing from ${LLM_CACHE_DIR}"
    "$PY" -u fill_llm_cache.py --llm-cache-dir "$LLM_CACHE_DIR" --generation-batch-size 4 \
        2>&1 | tee "$LOG_DIR/fill_cache.log"
fi
echo "[$(stamp)] cache coverage check (GPU perception)"
"$PY" -u tools/check_cache_coverage.py --device cuda --llm-cache-dir "$LLM_CACHE_DIR" \
    --min-train 8847 --min-test 483 \
    2>&1 | tee "$LOG_DIR/cache_coverage.log"

# ---- Train -------------------------------------------------------------------
for S in $SEEDS; do
    MODEL_DIR="$CKPT_BASE/seed${S}_frozenhead"
    echo "[$(stamp)] training seed ${S} -> ${MODEL_DIR}"
    "$PY" -u minedit_llm_mask_calibrate.py \
        --data satnet \
        --seed "$S" \
        --epochs 20 \
        --batch-size 200 \
        --model-dir "$MODEL_DIR" \
        --llm-cache-dir "$LLM_CACHE_DIR" \
        --generation-batch-size 4 \
        --eval-mask-biases="-2.0,-1.0,0.0,1.0,2.0" \
        --minedit-iters-eval 10 \
        --skip-cache-miss \
        --use-perception-features \
        --temp-hidden-dim 128 \
        --lambda-mask 1.0 \
        --lambda-temp 0.5 \
        --label-smoothing 0.3 \
        --stage-mask-freeze-epochs 0 \
        --patience 7 \
        --edit-layer casper \
        --casper-group-solver "$GROUP_SOLVER" \
        --freeze-fusion-head \
        ${EXTRA_ARGS} \
        2>&1 | tee "$LOG_DIR/train_seed${S}_frozenhead.log"
done

# ---- Test evaluation -----------------------------------------------------------
echo "[$(stamp)] evaluating seeds ${SEEDS_CSV} on -test"
"$PY" -u evaluate_checkpoints.py \
    --folder "$CKPT_BASE" \
    --seed-subdir-pattern "seed{seed}_frozenhead" \
    --seeds "$SEEDS_CSV" \
    --minedit --llm \
    --edit-layer casper \
    --casper-group-solver "$GROUP_SOLVER" \
    --llm-cache-dir "$LLM_CACHE_DIR" \
    --skip-cache-miss \
    --generation-batch-size 4 \
    --python-bin "$PY" \
    --output-tag frozenhead \
    2>&1 | tee "$LOG_DIR/eval_seeds_${SEEDS_CSV//,/_}_frozenhead.log"

echo "[$(stamp)] DONE. Summary: ${CKPT_BASE}/evaluation_summary__frozenhead.log"
