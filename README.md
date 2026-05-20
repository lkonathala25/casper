# MNIST-Sum and MNIST-Add with closed-form constraint projection

Reference implementation of the MinEdit L2 projection layer applied to two
neuro-symbolic benchmarks:

* **MNIST-Sum**: predict the sum of `N` MNIST digits.
* **MNIST-Add**: predict the place-value sum of two `N`-digit MNIST numbers
  using a carry-chain decomposition.

Both tasks share a LeNet-5 digit encoder. The constraint layer projects the
predicted output distribution onto the constraint surface defined by the
input-digit argmaxes — closed form by default, with a CVXPY/SCS QP backend
shipped as a numerical reference and scaling baseline.

## Layout

```
.
├── README.md
├── requirements.txt
├── run_mnist_sum.sh                     # MNIST-Sum sweep
├── run_mnist_add.sh                     # MNIST-Add sweep
└── src/
    ├── minedit.py                       # Closed-form L2 MinEdit projection (primitive)
    ├── cvxpy_minedit.py                 # CVXPY/SCS QP reference (drop-in for minedit)
    ├── constraint_layer.py              # MNIST-Sum constraint layer (wraps minedit)
    ├── carry_chain_constraint_layer.py  # MNIST-Add constraint layer (wraps minedit)
    ├── model.py                         # LeNet-5 backbone + sum/carry-chain heads
    ├── data.py                          # MNIST-Sum and MNIST-Add datasets
    ├── loss.py                          # Decomposed sum-task loss
    └── train.py                         # Training loop entry point
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`torch` is pinned at `>=2.0` only; install the build matching your CUDA
version per the official PyTorch instructions if you want GPU acceleration.

## Running the experiments

Both bash scripts sweep `N x seeds x backends` and write one CSV per
`(N, backend)` to `logs/<task>/`. The CSV columns are:

```
task, n_digits, projection_backend, run_id, epoch, loss,
neural_acc, symbolic_acc, constraint_sat,
avg_digit_acc, avg_l2_edit, grad_norm,
gpu_mem_mb, epoch_time_s
```

A NaN row at `epoch < requested_epochs` indicates the run hit the per-iter or
per-run wall-clock cap (used to short-circuit the cvxpy backend at large `N`).

### MNIST-Sum

```bash
bash run_mnist_sum.sh
```

* `N` in `{2, 16, 64, 256, 1024}`
* 5 seeds (0 through 4)
* batch size 16, 200 epochs
* Adam lr `1e-3`, cosine schedule with `eta_min=1e-5`
* 5000 train / 1000 test samples per `N`

### MNIST-Add

```bash
bash run_mnist_add.sh
```

* `N` in `{2, 15, 100, 256}`
* 5 seeds (0 through 4)
* batch size 64, 100 epochs
* Adam lr `1e-3` (`3e-4` for `N` in `{100, 256}`), cosine schedule with `eta_min=1e-5`
* `floor(60000 / 2N)` train / `floor(10000 / 2N)` test samples per `N`

The carry-chain decomposition is used for every `N` — there is no fallback to
a flat sum head.

### Single-run invocation

The bash scripts are thin loops over `src/train.py`. To run a single
`(task, N, seed, backend)`:

```bash
python src/train.py \
    --task sum --n-digits 16 --seed 0 \
    --projection-backend closed_form \
    --csv logs/mnist_sum/sum_n16_closed_form.csv
```

`--epochs`, `--batch-size`, and `--lr` use the spec defaults if omitted; pass
them explicitly to override.

## Reproducing the paper

The `(N, seed, backend)` CSVs feed mean ± std reduction over the 5 seeds for
each `(N, backend)` cell. Closed-form rows are the production results;
cvxpy-backend rows demonstrate that the QP-based projection does not scale at
large `N` (per-iter / per-run wall-clock caps fire and CSV rows become NaN).

## Key environment variables (bash scripts)

| Variable | Default | Effect |
|----------|---------|--------|
| `EPOCHS` | task-default | Epochs per run. |
| `BATCH_SIZE` | task-default | Mini-batch size. |
| `SOLVER_TIME_LIMIT` | `10` | SCS `time_limit_secs` per QP solve (cvxpy only). |
| `PER_ITER_TIMEOUT_S` | `30` | Wall-clock cap per training step. |
| `MAX_RUN_TIME_S` | `3600` | Wall-clock cap per `(N, seed, backend)` run. |

These exist to keep the cvxpy backend bounded at large `N`; they do not
affect the closed-form path under realistic budgets.
