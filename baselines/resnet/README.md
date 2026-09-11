# Unconstrained ResNet-18 (Warcraft)

No ported code. This is CASPER's own trainer, `tasks/warcraft/train.py`, run with
`--condition no_crl --constraint-mode none --alphabet full`: the same dataset,
backbone class, per-cell categorical head (868 outputs at 12x12), per-cell CE loss
and metrics as CASPER, with no constraint layer built or applied.

It replaces upstream `CTSketch/warcraft_12x12_baseline/train_unconstrained.py`, whose
loss is numerically identical but which used the monotone-only category fork.

**Run:** `run_warcraft.sh` (`GRIDS="12 30"` for both sizes).
