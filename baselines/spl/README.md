# SPL (Semantic Probabilistic Layers)

Ported from `/home/lk1325/SPL` (github.com/KareemYousrii/SPL).

**The real method is `train_mnistsum_spl.py`** -- pre-compiled SDD plus a
semantic loss. Note that both upstream run scripts (`run_mnistadd_n{2,4}.sh`)
invoke `train_mnistadd_supervised.py`, which is a plain supervised baseline with
no SDD and no SPL layer. The only successful SPL run in the source tree is
`mnistsum_n2.log`, which begins "Loaded SDD from sdd_files/mnistsum_N2.sdd";
`mnistadd_n2.log` contains a header and no epochs. This matches the published
table, where SPL appears at n=2 on Sum only.

**Included:** the SPL and supervised trainers, `compile_*_sdd.py`,
`verify_*_sdd.py`, `cutils/`, `grids/pypsdd` (a vendored pypsdd that supports
`log_space` in `generate_tf_ac`), and `sdd_files/mnistsum_N2.{sdd,vtree}`.

**Not included:** upstream's 13 GB, essentially all Warcraft data. SPL's MNIST
path needs only `pysdd`, `torch` and `torchvision` -- not the 2023
TensorFlow-era environment its `environment.yml` describes.

**Adapted:** MNIST resolves to the repo-root `data/` (was a bare relative
`"data"`).

Only `mnistsum_N2` is compiled. Larger N requires `compile_mnistsum_sdd.py`
first, which is where an exact method stops being tractable.
