# Baselines

One directory per method, each self-contained:

```
<method>/
  <task dirs or package>   vendored + adapted code
  run_sum.sh  run_add.sh   canonical invocations
  README.md                provenance and how it was ported
```

All baselines read MNIST from the repo-root `data/` — never a private copy.

## Environment

`ctsketch` and `anesi` share **`casper_env`** with CASPER itself:

```bash
conda create -n casper_env python=3.10
pip install torch==2.6.0 torchvision==0.21.0 tensorly==0.9.0 \
            numpy scipy wandb tqdm pyyaml pytest cvxpy cvxpylayers
```

CASPER's own suite passes on this env (227 tests), so one interpreter covers
all three. Methods added later will not all fit here — Scallop needs a compiled
`scallopy` (Rust), DeepSoftLog needs a Cython build, and SPL pins a 2023
Python 3.9 + TensorFlow stack. Those get their own `env.yml`.

## Smoke tests (1 epoch, smallest n, `casper_env`)

| method | task | result |
|---|---|---|
| CTSketch | sum, n=2 | acc 80.00%, digit 89.25% |
| CTSketch | add, 2 digits | acc 85.92%, digit 96.23% |
| A-NeSI | sum, arity=2 | acc 80.56%, digit 89.98% |
| A-NeSI | add, N=2 | acc 29.49%, digit 88.82% |

Sanity only — the published n=2 figures are ~95-98% after 50-100 epochs.
