# Archived: MNIST on-the-fly / closed-form constraint path

Superseded by the compiled path (`src/compiled_layer.py` +
`src/functional_dag_compile.py`). Kept only until the results below no longer
need reproducing, then deletable in full.

| file | what it was |
|---|---|
| `mnist_sum_constraint_layer.py` | MNIST-Sum layer. Recomputed the constraint every forward pass: `expected_sum = self.constraint_fn(digit_preds)`. Was `src/constraint_layer.py`. |
| `carry_chain_constraint_layer.py` | MNIST-Add layer, same on-the-fly pattern over the carry chain. |
| `cvxpy_minedit.py` | CVXPY/SCS QP projection backend — the numerical reference and scaling baseline. |
| `mnist_masked_loss.py` | MNIST-Sum's two-term loss (`L_pos` through the projection, `L_neg` on the raw distribution). Needed because the flat layer reports its L2 edit as a float, not a tensor, so that path has no differentiable projection term. Was `src/loss.py:MaskedLoss`. |
| `addn_constraint.py` | `_make_addn_constraint`, the place-value addition target the two above consume. |

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

`src/` no longer references this directory at all — there is no `closed_form`
or `cvxpy` backend on the live path, and `src/train.py` always builds the
compiled layer. The only things that reach in here are the equivalence checks
that use these layers as oracles: `tests/test_compiled_mnist.py`,
`tests/test_casper_layer.py`, and `examples/benchmark_mnist_add_compile.py`.
