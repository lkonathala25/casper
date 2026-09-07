# A-NeSI

Ported from `/home/lk1325/CTSketch/anesi` (github.com/HEmile/a-nesi), copied
whole — it is self-contained (`anesi.py`, `inference_models.py`,
`fit_dirichlet.py`, `util.py`, plus `experiments/`).

**Adapted:** `_DATA_ROOT` in `experiments/mnist_op/data/MNIST.py` now resolves
to the repo-root `data/`. Nothing else changed.

Both tasks use the same entry point, selected by config:

```yaml
# sum                  # add
N: 1                   N: <digits per operand>
arity: <n digits>      arity: 2
                       op: add_n
```

Run with `python -m experiments.mnist_op.mnistop --config <file>` from this
directory; `run_sum.sh` / `run_add.sh` generate the configs and sweep.
