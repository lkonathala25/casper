# DeepSoftLog

Ported from `/home/lk1325/CTSketch/deepsoftlog` (github.com/jjcmoon/DeepSoftLog).

**Excluded:** `experiments/mnist_addition/tmp` (64 MB scratch) and the darwin /
cp311 / cp313 builds of the Cython extensions.

**No build step needed.** The four Cython extensions ship pre-built for
CPython 3.10 Linux, which is what `casper_env` is:

```
algebraic_prover/terms/{expression,variable,probability_annotation}.cpython-310-x86_64-linux-gnu.so
algebraic_prover/proving/proof_queue.cpython-310-x86_64-linux-gnu.so
```

Rebuilding on another Python needs `python setup.py build_ext --inplace`
with Cython.

**Adapted:** both `experiments/mnist_{sum,addition}/dataset.py` now resolve
MNIST to the repo-root `data/`. `mnist_addition` previously downloaded into the
excluded `tmp/` directory.

**Running:** needs `PYTHONPATH=.` from this directory -- invoking
`deepsoftlog/run_experiments.py` directly puts the script's own directory on the
path rather than the package parent. `run_sum.sh` handles this.

## Warcraft (`deepsoftlog/experiments/warcraft/`)

Ported from `/home/lk1325/CTSketch/deepsoftlog/experiments/warcraft/`, the trainer
behind the upstream 2-seed results (runner `bash_scripts/warcraft/run_dsl_warcraft.sh`).
`embeddings/nn_models.py` (`ResNetNode`, `WarcraftPathFunctor`) was already here and
is byte-identical to upstream.

**Adapted:** dataset, per-cell categories, loss and metrics come from `tasks/warcraft/`
(the upstream fork was monotone-only, with silent `skip` relabelling). The full
alphabet (868) is set by `dataset.ALPHABET`. `config.yaml` points `data_dir` at
`data/warcraft/12x12/preprocessed_4conn` and drops the `-DELETE` run name.

**Constraint: steps encoding (`constraint: steps` in `config.yaml`, the default).**
`WarcraftPathFunctor` now computes P(valid) over the time-unrolled walk X_0..X_L on
the compiled domains D_i from CASPER's steps artifact, with edge probabilities as
move weights and an absorbing sink. The original version ran `num_path_steps: 22`
free relaxations, so only 22-edge (minimum-length, i.e. monotone) paths could reach
the sink: on ground-truth inputs it gave P(valid) = 1 on 0 of 57 non-monotone test
boards, against 57 of 57 for the steps version. `constraint: reach` keeps it.

`num_path_steps` is used, and logged, only in `reach` mode.

**Batch norm:** each prover query runs `ResNetNode` on a single image in train mode,
which updated batch-norm running statistics once per query -- 32 updates per step
against one from the batched CE forward. Evaluation normalises with those running
statistics; measured on Warcraft images, after 6 steps that shifted eval-mode logits
by a mean of 0.357 against a logit std of 0.171. `WarcraftTrainer` now freezes
running-statistic updates during the prover branch (`frozen_bn_stats`), so they come
from batched forwards only; that branch's own normalisation and gradients are
unchanged.

Still to note: the loss scores walks, so it does not by itself forbid extra edges
off the walk (the per-cell CE term discourages them). The path term is a training
loss only; inference is per-cell argmax.

**Run:** `run_warcraft.sh` (seeds 42 43, 100 epochs), from this directory with
`PYTHONPATH=.`.
