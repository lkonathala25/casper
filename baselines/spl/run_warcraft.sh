#!/usr/bin/env bash
# SPL on Warcraft 12x12: per-edge head (264 outputs x num_reps) feeding the
# compiled path circuit in constraints/warcraft/spl/ (FFCircuit trainer).
set -euo pipefail
cd "$(dirname "$0")/warcraft"
PY="${PY:-$HOME/miniconda3/envs/casper_env/bin/python}"
SEEDS="${SEEDS:-1 2 3}"; CFG="settings/warcraft_shortest_path/spl/12x12.json"
LOGDIR="${LOGDIR:-../../../logs/baselines}"; mkdir -p "$LOGDIR"
for seed in $SEEDS; do
  echo ">>> SPL warcraft 12x12 seed=$seed"
  PARAMS="{'default_json': '$CFG', 'seed': $seed, 'model_dir': 'results/warcraft_12x12_spl/seed$seed'}"
  "$PY" main.py "$PARAMS" 2>&1 | tee "$LOGDIR/spl_warcraft_12_s${seed}.log"
done
