# Warcraft Shortest Path with closed-form constraint projection

Reference implementation of the MinEdit L2 projection layer applied to the
Warcraft Shortest-Path benchmark (Pogancic et al. 2020) at two grid sizes,
12x12 and 30x30.

A ResNet-18 backbone (trained from scratch) predicts a per-node categorical
distribution over routing choices (skip / in→out pairs). The constraint layer
projects each node distribution onto the valid-path manifold, with the
projection used both as a training-time loss term and at inference. An
auxiliary L2 self-distillation term pulls the unprojected distribution toward
the projected one.

## Layout

```
.
├── README.md
├── requirements.txt
├── run_warcraft_12x12.sh                # Full method (\layer{}) — 12x12, 5 seeds
├── run_warcraft_30x30.sh                # Full method (\layer{}) — 30x30, 5 seeds
├── run_warcraft_ablation_12x12.sh       # Pure MinEdit ablation — 12x12, 3 conditions x 5 seeds
├── run_warcraft_ablation_30x30.sh       # Pure MinEdit ablation — 30x30, 3 conditions x 5 seeds
└── src/
    ├── minedit.py                       # Closed-form L2 MinEdit primitive
    ├── constraint_layer.py              # Per-node WarcraftConstraintLayer
    ├── masked_loss.py                   # MinEdit positive + negative head loss
    ├── model.py                         # ResNet-18 backbone + per-node head + metrics
    ├── data.py                          # WarcraftDataset
    ├── node_categories.py               # Variable layout / edge-pattern bookkeeping
    ├── train.py                         # Training loop entry point
    ├── compute_metrics.py               # Single/multi-checkpoint full-metric evaluation
    ├── compute_cost_ratio.py            # Cost-ratio metric utilities (library)
    └── cost_ratio.py                    # Cost-ratio multi-seed evaluation entry point
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

The Warcraft Shortest-Path dataset is from Pogancic et al. 2020
("Differentiation of Blackbox Combinatorial Solvers"). The expected layout is

```
data/warcraft_shortest_path_oneskin/
├── 12x12/preprocessed_4conn/
│   ├── train_maps.npy   train_edge_labels.npy   train_costs.npy   train_node_labels.npy   train_vertex_weights.npy
│   ├── val_maps.npy     val_edge_labels.npy     val_costs.npy     val_node_labels.npy     val_vertex_weights.npy
│   ├── test_maps.npy    test_edge_labels.npy    test_costs.npy    test_node_labels.npy    test_vertex_weights.npy
│   ├── edges.npy
│   └── node_to_edges.npz
└── 30x30/preprocessed_4conn/
    └── (same file layout as 12x12)
```

Both bash scripts default to `DATA_DIR=data/warcraft_shortest_path_oneskin/{12x12,30x30}/preprocessed_4conn`;
override via the `DATA_DIR` environment variable if your dataset lives
elsewhere.

## Running the experiments

Each script runs `\layer{}` (the full method) over 5 seeds with the paper's
training recipe. Per-seed checkpoints, CSVs, and logs are written to
`logs/warcraft_{12x12,30x30}/seed{0..4}/`.

### 12x12

```bash
bash run_warcraft_12x12.sh
```

* Grid: 12x12
* 5 seeds (0 through 4)
* ResNet-18 trained from scratch
* Adam lr 5e-4, cosine schedule (eta_min = 5e-6), batch 128
* 300 epochs with early stopping (patience 30)
* MinEdit positive + negative head loss with surrogate threshold eps = 1e-3
* Auxiliary L2 self-distillation, lambda = 1.0
* Variables processed in row-major order (set `PROCESSING_ORDER=diagonal`
  for the alternative; empirically equivalent on this task)

### 30x30

```bash
bash run_warcraft_30x30.sh
```

* Grid: 30x30
* 5 seeds (0 through 4)
* Same backbone / optimiser / schedule / batch as 12x12
* 300 epochs with early stopping (patience 50)
* MinEdit + L2 self-distillation lambda = 1.0
* Variables processed in row-major order

## Ablations

The two ablation scripts run a pure-MinEdit ablation: the L2 self-distillation
term and every auxiliary loss are disabled, so each row isolates the effect
of where the projection is applied:

| Condition (`--condition`) | Training loss | Eval projection |
|---|---|---|
| `crl_in_training_eval_off` | `WarcraftMaskedLoss` (MinEdit pos+neg) | OFF |
| `crl_inference_only` | uniform per-node CE | ON |
| `crl_in_training` | `WarcraftMaskedLoss` (MinEdit pos+neg) | ON |

```bash
bash run_warcraft_ablation_12x12.sh
bash run_warcraft_ablation_30x30.sh
```

Outputs land in `logs/warcraft_ablation_{12x12,30x30}/<condition>/seed{0..4}/`.

## Metrics and post-hoc evaluation

The training script writes a per-epoch CSV with `exact`, `hamming`,
`consistent` columns (raw and post-projection). For the full set of metrics
reported in the paper (path-level + cost-optimality at multiple percentiles)
use `compute_metrics.py`, which works against either a single checkpoint or a
directory of checkpoints and supports both projected and raw inference modes:

```bash
# Aggregate across all 5 seeds with the constraint layer applied at inference
python src/compute_metrics.py \
    --ckpt-dir logs/warcraft_12x12 \
    --data_dir data/warcraft_shortest_path_oneskin/12x12/preprocessed_4conn \
    --grid-size 12 \
    --processing-order row \
    --apply-constraints true \
    --out_json logs/warcraft_12x12/metrics.json

# Same checkpoints, evaluated with the constraint layer DISABLED (raw output)
python src/compute_metrics.py \
    --ckpt-dir logs/warcraft_12x12 \
    --data_dir data/warcraft_shortest_path_oneskin/12x12/preprocessed_4conn \
    --grid-size 12 \
    --apply-constraints false \
    --out_json logs/warcraft_12x12/metrics_eval_off.json

# Single checkpoint
python src/compute_metrics.py \
    --ckpt logs/warcraft_12x12/seed0/crl_in_training/best_run0.pt \
    --data_dir data/warcraft_shortest_path_oneskin/12x12/preprocessed_4conn \
    --grid-size 12
```

The output JSON contains per-checkpoint metrics and (when more than one
checkpoint is provided) a `summary` block with mean ± std for every metric:

```
exact, hamming, consistent,
ratio_eq_1.0, ratio_le_1.01, ratio_le_1.05, ratio_le_1.10,
ratio_mean_on_valid, ratio_median_on_valid, valid_pct
```

`cost_ratio.py` is the older cost-ratio-only entry point and is kept for
backwards compatibility; new analyses should use `compute_metrics.py`.

## Notes on key environment variables

| Variable | Default (12x12 / 30x30) | Effect |
|----------|-------------------------|--------|
| `EPOCHS` | 300 / 300 | Epochs per seed. |
| `PATIENCE` | 30 / 50 | Early-stopping patience on val_exact. |
| `BATCH_SIZE` | 128 | Mini-batch size. |
| `LR` | 5e-4 | Adam learning rate. |
| `LAMBDA_L2_EDIT` | 1.0 | Weight on L2 self-distillation aux loss. |
| `PROCESSING_ORDER` | `row` | Variable update order for the projection (`row` or `diagonal`). |
| `DATA_DIR` | `data/warcraft_shortest_path_oneskin/<NxN>/preprocessed_4conn` | Path to the preprocessed dataset. |
