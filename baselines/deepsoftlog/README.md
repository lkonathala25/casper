# DeepSoftLog

Ported from `/home/lk1325/CTSketch/deepsoftlog` (github.com/jjcmoon/DeepSoftLog).

**Excluded:** `experiments/mnist_addition/tmp` (64 MB scratch), the Warcraft
experiment, and the darwin / cp311 / cp313 builds of the Cython extensions.

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
