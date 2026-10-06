#!/usr/bin/env bash
# GPU-only parity gates of the casper port (CASPER_MASK_TRAINING_PORT.md §7.2).
# Each gate prints PASS or FAIL; the script stops at the first FAIL.
#
#   P1  Transformer CASPER reference ckpt, seed 0   NeuroAcc 62.0  RecRate 93.26599  PipeAcc 82.6  certified 317
#   P2  LLM frozen-head reference ckpts, seeds 0,1  62.0/93.92936/80.0 and 62.0/93.44484/81.0; cache hits only
#   P3  Legacy MinEdit reference ckpt, seed 0       NeuroAcc 66.0  PipeAcc 84.8 (--edit-layer minedit)
#   P4  run_mask_calibrated.sh smoke (--epochs 1)   epoch-0 line == reference log
#   P5  run_llm_mask_calibrated.sh smoke (--epochs 1) epoch-0 line == reference log; head FROZEN, 1797505 params
#
# Reference checkpoints are COPIED into $CKPT_DIR/parity/ and evaluated there;
# nothing is written into REFERENCE_CKPT_DIR (llm-sudoku) or the LLM cache.
#
# Usage (from tasks/sudoku/visual; detached):
#   mkdir -p results/gpu_gates
#   GPU_ID=0 setsid nohup scripts/verify_gpu_gates.sh > results/gpu_gates/driver.log 2>&1 < /dev/null &
#
# Env:
#   GPU_ID          GPU to use (default 0)
#   PY_SUDOKU       python of the `sudoku` env (default <conda envs dir>/sudoku/bin/python,
#                   e.g. /home/la2324/miniconda3/envs/sudoku/bin/python)
#   PY_SUDOKU_FINE  python of the `sudoku-fine` env (default <conda envs dir>/sudoku-fine/bin/python)
#   GATES           space-separated subset to run (default "P1 P2 P3 P4 P5")
set -euo pipefail

VISUAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$VISUAL_DIR"
eval "$(python3 paths.py --shell)"

GPU="${GPU_ID:-0}"
ENVS_DIR="$(dirname "$SWIPL_ENV")"
PY_T="${PY_SUDOKU:-$ENVS_DIR/sudoku/bin/python}"
PY_L="${PY_SUDOKU_FINE:-$ENVS_DIR/sudoku-fine/bin/python}"
GATES="${GATES:-P1 P2 P3 P4 P5}"
PARITY="$CKPT_DIR/parity"
LOG_DIR="$RESULTS_DIR/gpu_gates"
CASPER_RUN="e10_lr1e-4_wd3e-1_bs200_cos_wu0_nw20_clip1_it3-10_pat5_d1_bs0_-1_-2_casper"
LEGACY_RUN="e10_lr1e-4_wd3e-1_bs200_cos_wu0_nw20_clip1_it3-10_pat5_d1_bs0_-1_-2"

mkdir -p "$PARITY" "$LOG_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
pass() { echo "[$(stamp)] [gate] $1 PASS"; }
fail() { echo "[$(stamp)] [gate] $1 FAIL: $2"; exit 1; }
want() { [[ " $GATES " == *" $1 "* ]]; }

# copy_ckpt <src dir> <dst dir> <seed>: checkpoint + sidecar only.
copy_ckpt() {
    mkdir -p "$2"
    cp "$1/checkpoint_best_R_$3.pth" "$1/checkpoint_best_R_$3.json" "$2/"
}

# check_csv <gate> <csv> <seed> key=expected ...: float compare at the expected's precision.
check_csv() {
    local gate="$1" csv="$2" seed="$3"; shift 3
    python3 - "$csv" "$seed" "$@" <<'EOF' || fail "$gate" "metrics differ (see above)"
import csv, sys
path, seed, checks = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = [r for r in csv.DictReader(open(path)) if r['seed'] == seed]
assert len(rows) == 1, f'seed {seed} not found once in {path}'
ok = True
for chk in checks:
    key, exp = chk.split('=')
    got = float(rows[0][key])
    places = len(exp.split('.')[1]) if '.' in exp else 0
    good = round(got, places) == float(exp)
    ok &= good
    print(f'  seed {seed} {key}: got {got!r} expected {exp} -> {"ok" if good else "MISMATCH"}')
sys.exit(0 if ok else 1)
EOF
}

# check_line <gate> <log> <expected exact line>
check_line() {
    if grep -qxF -- "$3" "$2"; then
        echo "  found: $3"
    else
        echo "  expected line not found in $2:"
        echo "    $3"
        echo "  got:"
        grep -F -- "${3%%]*}]" "$2" | sed 's/^/    /' || true
        fail "$1" "epoch-0 line differs"
    fi
}

echo "[$(stamp)] gates='${GATES}' gpu=${GPU} sudoku=${PY_T} sudoku-fine=${PY_L}"
echo "[$(stamp)] parity dir ${PARITY}; logs ${LOG_DIR}"

# ---- P1: Transformer, CASPER ---------------------------------------------------
if want P1; then
    DST="$PARITY/transformer_casper"
    copy_ckpt "$REFERENCE_CKPT_DIR/casper/mask_calibrated/$CASPER_RUN" "$DST" 0
    "$PY_T" -u evaluate_checkpoints.py --folder "$DST" --seeds 0 --minedit \
        --edit-layer casper --python-bin "$PY_T" --output-tag port_parity \
        > "$LOG_DIR/P1.log" 2>&1 || fail P1 "evaluate_checkpoints.py failed (see $LOG_DIR/P1.log)"
    check_csv P1 "$DST/evaluation_per_seed__port_parity.csv" 0 \
        neuro_solver_board_acc_pct=62.0 perception_corr_solver_pct=93.26599 \
        nasr_board_acc_pct=82.6 casper_certified=317
    pass P1
fi

# ---- P2: LLM, frozen fusion head, CASPER -----------------------------------------
if want P2; then
    DST="$PARITY/llm_frozenhead"
    for S in 0 1; do
        copy_ckpt "$REFERENCE_CKPT_DIR/casper/llm_mask_calibrated/seed${S}_frozenhead" \
            "$DST/seed${S}_frozenhead" "$S"
    done
    "$PY_L" -u evaluate_checkpoints.py --folder "$DST" \
        --seed-subdir-pattern "seed{seed}_frozenhead" --seeds 0,1 --minedit --llm \
        --edit-layer casper --llm-cache-dir "$LLM_CACHE_DIR" --skip-cache-miss \
        --generation-batch-size 4 --python-bin "$PY_L" --output-tag port_parity \
        > "$LOG_DIR/P2.log" 2>&1 || fail P2 "evaluate_checkpoints.py failed (see $LOG_DIR/P2.log)"
    check_csv P2 "$DST/evaluation_per_seed__port_parity.csv" 0 \
        neuro_solver_board_acc_pct=62.0 perception_corr_solver_pct=93.92936 nasr_board_acc_pct=80.0
    check_csv P2 "$DST/evaluation_per_seed__port_parity.csv" 1 \
        neuro_solver_board_acc_pct=62.0 perception_corr_solver_pct=93.44484 nasr_board_acc_pct=81.0
    STATS="[rl_eval_sudoku] final LLM adapter stats: parse_fail_count=0 strict_fail_count=0 fallback_count=0 cache_hits=483 cache_misses=17"
    N_STATS="$(grep -cxF -- "$STATS" "$DST/evaluation_raw__port_parity.log" || true)"
    echo "  adapter stats line (cache only, 17 uncached zero-logit boards) seen ${N_STATS}/2 times"
    [[ "$N_STATS" == "2" ]] || fail P2 "adapter stats differ from: $STATS"
    pass P2
fi

# ---- P3: legacy MinEdit (Hungarian) ----------------------------------------------
if want P3; then
    DST="$PARITY/legacy_minedit"
    copy_ckpt "$REFERENCE_CKPT_DIR/minedit/mask_calibrated/$LEGACY_RUN" "$DST" 0
    "$PY_T" -u evaluate_checkpoints.py --folder "$DST" --seeds 0 --minedit \
        --edit-layer minedit --python-bin "$PY_T" --output-tag port_parity \
        > "$LOG_DIR/P3.log" 2>&1 || fail P3 "evaluate_checkpoints.py failed (see $LOG_DIR/P3.log)"
    check_csv P3 "$DST/evaluation_per_seed__port_parity.csv" 0 \
        neuro_solver_board_acc_pct=66.0 nasr_board_acc_pct=84.8
    pass P3
fi

# ---- P4: Transformer retrain smoke ---------------------------------------------
if want P4; then
    SMOKE="$PARITY/smoke_transformer"
    SUDOKU_CKPT_DIR="$SMOKE/checkpoint" SUDOKU_RESULTS_DIR="$SMOKE/results" \
    SEEDS=0 GPU_ID="$GPU" PYTHON_BIN="$PY_T" EXTRA_ARGS="--epochs 1" \
        scripts/run_mask_calibrated.sh > "$LOG_DIR/P4.log" 2>&1 \
        || fail P4 "run_mask_calibrated.sh failed (see $LOG_DIR/P4.log)"
    check_line P4 "$SMOKE/results/mask_calibrated/train_seed0.log" \
        "----[mask-calib][Epoch 0] best_bias=-2.000 board_acc=0.7660 (383/500) cell_acc=0.9855 | neuro_board=291/500 wrong_corrected=92/209 (0.4402) corrupted_correct=0/291 (0.0000) | blank_rate=0.0535 keep_rate_wrong_proj=0.2774 | alldiff_sat=0.9474"
    pass P4
fi

# ---- P5: LLM retrain smoke -------------------------------------------------------
if want P5; then
    SMOKE="$PARITY/smoke_llm"
    SUDOKU_CKPT_DIR="$SMOKE/checkpoint" SUDOKU_RESULTS_DIR="$SMOKE/results" \
    SEEDS=0 GPU_ID="$GPU" PYTHON_BIN="$PY_L" EXTRA_ARGS="--epochs 1" \
        scripts/run_llm_mask_calibrated.sh > "$LOG_DIR/P5.log" 2>&1 \
        || fail P5 "run_llm_mask_calibrated.sh failed (see $LOG_DIR/P5.log)"
    LOG="$SMOKE/results/llm_mask_calibrated/train_seed0_frozenhead.log"
    check_line P5 "$LOG" "[llm-calib] fusion head FROZEN; 1797505 trainable params"
    check_line P5 "$LOG" \
        "----[llm-calib][Epoch 0] (s) solver_neuro=0.5804 (285/491) | (a) llm_raw=0.1405 (69/491) | (b) fused_neuro=0.5804 (285/491) | (c) board_acc=0.7617 (374/491) @ bias=-2.000 cell_acc=0.9853 wrong_corrected=89/206 (0.4320) corrupted_correct=0/285 (0.0000) | blank_rate=0.0535 keep_rate_wrong_proj=0.2826 | alldiff_sat=0.9478"
    pass P5
fi

echo "[$(stamp)] all requested gates PASSED: ${GATES}"
