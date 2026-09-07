# Baselines

One directory per method. Each holds:

```
<method>/
  src/       vendored + adapted code (small; datasets are not copied here)
  env.yml    locked interpreter for this method
  run.sh     canonical invocation, naming its own python
```

They cannot share one interpreter — Scallop needs a compiled `scallopy` (Rust),
DeepSoftLog needs a Cython build, and SPL pins a 2023 Python 3.9 + TensorFlow
stack. So "one place" means one repo and one command surface, not one env.

For MNIST specifically, most methods do fit a single environment: blackbox,
ICR, ISED, CCN+ and DeepProbLog all run on the `dpl` interpreter as-is, and
CTSketch needs only `tensorly` plus a vendored `tt_sketch`. Scallop and SPL are
the two that genuinely need isolation.

All methods read MNIST from `data/` at the repo root -- never a private copy.
