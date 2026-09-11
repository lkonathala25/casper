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

## Warcraft (`warcraft/`)

Ported from `/home/lk1325/SPL/WarcraftShortestPath/`, code only (no data,
checkpoints, results or logs). The compiled path circuit
`constraint_trimmed.{sdd,vtree}` (12.8 MB) lives in `constraints/warcraft/spl/`.
`grids/` is shared with the MNIST port (identical).

**Adapted:** the circuit, `e2i.pt` and `grids/` are resolved from the file location
rather than the working directory. The loader reads this repo's `*_edge_labels.npy`
when `*_shortest_paths.npy` is absent (verified identical arrays). All settings point
at `data/warcraft/12x12/preprocessed_4conn`. `loader_params.train_limit` was added for
smoke tests. `settings/warcraft_shortest_path/spl/12x12.json` is the FFCircuit (SPL
proper) configuration, taken from upstream's
`results/warcraft_shortest_path_baseline/settings.json`. Added to `casper_env`:
`tensorboardX`, `psutil`, `pathlib2`.

**Head:** per-edge, `264 x num_reps + num_reps` outputs -- not the per-cell
categorical head the other methods use. The ResNet-18 body matches, but `conv1` is
rebuilt (different initialisation) and images are mean/std standardised.

**Before comparing:**
- The circuit is `paths.smaller(33)` (`data/compile_paths.py`): simple s-t paths of
  at most 32 edges, about 3e10 models. No ground truth exceeds 32 edges, but it is a
  prior the other methods don't get, and there is no 30x30 circuit. Rebuilding needs
  `graphillion` and the `sdd` bindings, which are not installed.
- `use_test_set: true`: upstream selects its checkpoint on the **test** split.
- `neighbourhood_fn: 8-grid` is passed to SPL's Dijkstra module while edges and
  labels are 4-connected; not yet traced whether it affects reported metrics.
- `trainers.py` also holds `SLTrainer` (semantic loss) and `BaselineTrainer`; only
  `FFCircuit` / `Circuit` are SPL.

**Run:** `run_warcraft.sh` (seeds 1 2 3).
