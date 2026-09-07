# Scallop

Ported from `/home/lk1325/CTSketch/{mnist-sum,mnist-add}` (scallop-lang.org).

| here | upstream |
|---|---|
| `mnist_sum/` | `mnist-sum/{scallop_sum,create_scallop_sum,mnist_config}.py` |
| `mnist_add/` | `mnist-add/{scallop_add,add_config}.py` |

**Paper setting:** top-3 proofs — `--top-k 3` (sum) / `--k 3` (add), provenance
`difftopkproofs`. Both are the upstream defaults.

**Adapted:** MNIST now resolves to the repo-root `data/`. `add_config.py` used a
bare relative `"data"`, which silently depended on the working directory.

**Dependency note:** `scallopy` 0.2.5 is a compiled Rust extension with no
published wheel here. It was transplanted into `casper_env` from the `ctsketch`
env, which works only because both are CPython 3.10
(`scallopy.cpython-310-x86_64-linux-gnu.so`). If `casper_env` is ever rebuilt on
a different Python, this has to be rebuilt from Rust source.
