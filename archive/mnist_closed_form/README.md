# Archived: MNIST on-the-fly / closed-form constraint path

Superseded by the compiled path (`src/compiled_layer.py` +
`src/functional_dag_compile.py`). Kept only until the results below no longer
need reproducing, then deletable in full.

| file | what it was |
|---|---|
| `mnist_sum_constraint_layer.py` | MNIST-Sum layer. Recomputed the constraint every forward pass: `expected_sum = self.constraint_fn(digit_preds)`. Was `src/constraint_layer.py`. |
| `carry_chain_constraint_layer.py` | MNIST-Add layer, same on-the-fly pattern over the carry chain. |
| `cvxpy_minedit.py` | CVXPY/SCS QP projection backend — the numerical reference and scaling baseline. |

`compute_carry_chain` was **not** archived: it is task arithmetic (the
ground-truth sum digits) used for a consistency metric, and now lives in
`src/carry_chain.py`.

## Why it is still here

1. **Test oracles.** The compiled path is proved correct by comparing against
   these layers: `test_sum_n2_equals_hardcoded_layer`,
   `test_add_n2_equals_hardcoded_layer`, `test_add_infer_matches_compute_carry_chain`,
   `test_add_n3_generated_dsl_direct_load_equals_hardcoded`. Deleting this
   directory deletes that proof, so record the passing run first.
2. **The scaling figure.** `run_mnist_sum.sh` / `run_mnist_add.sh` sweep
   `closed_form` vs `cvxpy`; that contrast is the "QP projection does not
   scale" result. Those scripts pass `--projection-backend` explicitly and
   still work.
3. **MNIST-Sum at N > 2.** The flat sum encoding gives `S` a domain of 9N+1 and
   only N=2 compiles; larger N currently has no compiled path, so
   `--projection-backend closed_form` is the only way to run it.

Nothing on the compiled path imports from here. `src/model.py` appends this
directory to `sys.path` only when `closed_form` or `cvxpy` is requested.
