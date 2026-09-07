# CTSketch

Ported from `/home/lk1325/CTSketch` (github.com/alaiasolkobreslin/CTSketch).

| here | upstream |
|---|---|
| `mnist_sum/` | `mnist-sum/{cts_sum,mnist_config,sketch_configs}.py` |
| `mnist_add/` | `mnist-add/{cts_add,mnist_config,sketch_config}.py` |
| `tensor_sketch.py`, `tt_sketch/` | repo root |

Scallop files from those directories were not ported.

**Adapted:** `data_dir` now resolves to the repo-root `data/` instead of the
upstream repo's own copy. Nothing else changed.

**Paper setting:** TT rank 2 (`--rank 2`, `--method tt`).

`n` is not a direct flag. For **sum** it is `2**depth`, so `--depth 1 4 6 8 10`
gives n = 2, 16, 64, 256, 1024. For **add**, `--digits` is the operand width.

## Provenance caveat

A second, newer copy of `cts_sum.py` exists at
`mnist_experiment/CRL/mnist-sum/cts_sum.py` (2026-04-29, 267 lines) and differs
from the one ported here (2026-04-18, 339 lines). Neither results file matches
the published figure: `ctsketch_fully_supervised_final_results.csv` records
n=16 -> 84.0 and n=64 -> 48.4, while the figure says 82.35 and 51.65, and those
numbers are hard-coded in `mnist_experiment/mnist_sum_plotting.py`. The
published figures are therefore not reproducible from either file, which is why
this port is set up to produce fresh numbers rather than reproduce recorded ones.
