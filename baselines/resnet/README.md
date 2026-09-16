# Unconstrained ResNet-18 (Warcraft)

No ported code. This is CASPER's own trainer, `tasks/warcraft/train.py`, run with
`--condition no_crl --constraint-mode none --alphabet full`: the same dataset,
backbone class, per-cell categorical head (868 outputs at 12x12), per-cell CE loss
and metrics as CASPER, with no constraint layer built or applied.

It replaces upstream `CTSketch/warcraft_12x12_baseline/train_unconstrained.py`, whose
loss is numerically identical but which used the monotone-only category fork.

**Run:** `run_warcraft.sh` -- 5 seeds (0-4) x 300 epochs; `GRIDS="12 30"` for both sizes. Outputs go to `$RUNS_ROOT/warcraft/<N>x<N>/resnet/seed<S>/` (default `RUNS_ROOT=/Data/lk1325/runs`), log in `train.log`, `DONE` marker when finished so relaunches skip it; `DRY_RUN=1` only prints the commands (`tools/runlib.sh`).
