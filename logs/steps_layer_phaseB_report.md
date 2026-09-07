# Phase B — training integration

No training run. Two files changed, both additively:
`src_warcraft/model.py` (+49/-5), `src_warcraft/train.py` (+46/-1).

## Diff summary

### `model.py`

1. `constraint_mode='steps'` branch constructing `StepsProjectionLayer`
   (`src_steps` appended to `sys.path`, not prepended, per the repo's own
   convention). `certify=False` is the constructor default, so the
   training-time layer skips the independent validity check while inference
   keeps it.
2. New `certify` keyword, defaulting to `False`.
3. **The build-time dim assert**, after layer construction:

       layer_dim = getattr(self.constraint_layer, 'total_dim', None)
       if layer_dim != self.total_dim:
           raise ValueError(
               f"constraint layer total_dim {layer_dim} != head dim "
               f"{self.total_dim}: constraint_mode={constraint_mode!r} does "
               f"not match the alphabet the head was built for")

4. Docstring updated to name the third mode and the new invariant.

### `train.py`

1. New flags, all defaulting to today's behaviour:
   `--constraint-mode {monotone,lazy_exact,steps}` (default `monotone`),
   `--artifact` (default `None`), `--max-rounds` (default `20`). `--alphabet`
   already existed, default `monotone`, untouched.
2. Model construction now forwards `constraint_mode`, `alphabet`, `artifact`
   and `max_rounds`. Previously it forwarded none of them, which is what
   allowed the silent mismatch.
3. Startup gate: an explicit line reporting the label-conversion result per
   alphabet over all 12,000 instances. `WarcraftDataset` already routes through
   the fail-loud converter, so an unrepresentable pattern raises at startup;
   this makes the clean case visible instead of inferred from silence.
4. Config dump: `{save_dir}/config.json` with the full arg vector,
   `total_dim`, and the explicit `auxiliary_losses_off` list.

`--max-rounds` is retained only for `lazy_exact`; its help text states that the
steps layer has no repair loop, so the flag does not silently imply one.

## The assert, exercised

| alphabet | mode | head | layer | result |
|---|---|---|---|---|
| monotone | monotone | 628 | 628 | builds (**today's default, unchanged**) |
| full | monotone | 868 | 628 | **raises `ValueError`** — the silent-corruption case |
| full | steps | 868 | 868 | builds (the new path) |
| monotone | steps | 628 | 628 | builds |

The second row is the whole point: what previously returned a correctly-shaped
wrong answer now cannot be constructed.

One note on the fourth row: the dims match, so the assert passes, but the
reverse bridge would raise `UnrepresentablePatternError` the first time the
walk produced a non-monotone path. That is loud and correct, so I did not add
a second guard for a configuration we do not use — flagging it rather than
leaving it to be discovered.

## Hook and selection metric — matched, not re-decided

Unchanged from the original and verified by inspection at `train.py:224-226`
and `model.py`: the layer is fed `probs.detach()`, and the loss adds
`lambda_l2_edit * (probs - projected.detach()).pow(2).sum(dim=1).mean()`.
Stopped-gradient on both sides. `apply_constraints_in_train` remains
`lambda_l2_edit > 0.0`, so lambda = 0 runs no layer during training.

Selection stays `0.5*raw_exact + 0.5*proj_exact` under `crl_in_training`,
with `proj_exact` now flowing through the steps layer for both lambda configs,
so the delta is not confounded by a selection difference.

**End-to-end wiring smoke** (4 random images, one forward/backward, not a
training run): `logits/probs/projected` all `(4, 868)`;
`projected.requires_grad = False`; `masked = 1.8564`, `l2_edit = 2.0570`,
`loss = 3.9134`; `backbone.fc` gradient norm 12.82, all finite; layer reported
0 dead ends, 460 cells edited, mean arrival 22.0.

## Regression check

| suite | result |
|---|---|
| whole `tests/` in one process | **200 passed** |
| pre-existing tests alone (steps suites deselected) | **117 passed** |
| steps suites | **83 passed** |

117 + 83 = 200; nothing regressed and nothing is being skipped.

**One thing I broke and fixed.** Running the full suite first showed 3
failures in `tests/test_casper_layer.py`. They were not caused by the
`train.py`/`model.py` edit — they were caused by *my own steps test files*
from earlier phases prepending `src_warcraft` to `sys.path`. `src/` and
`src_warcraft/` both define `constraint_layer`, `data`, `model` and `minedit`,
so prepending shadowed `src/` for every test sharing the process. Confirmed by
running `test_casper_layer.py` alone (13 passed) and with one steps file added
(3 failed).

Fixed in `tests/_warcraft_import.py`: `load_warcraft(name)` prepends
`src_warcraft` only for the duration of the import and then evicts the
colliding module names from `sys.modules`, so a later `import constraint_layer`
still resolves against `src/`. The three steps test files use it. No
`src_warcraft` source file was touched to achieve this.

## Planned Phase C invocations

Common to all six runs:

    python src_warcraft/train.py \
      --data_dir /home/lk1325/mnist_experiment/warcraft_crl/\
warcraft_shortest_path_oneskin/12x12/preprocessed_4conn \
      --grid-size 12 \
      --alphabet full \
      --constraint-mode steps \
      --condition crl_in_training \
      --n-runs 1 \
      --epochs 300 --patience 30 \
      --batch-size 128 --lr 5e-4 --weight-decay 0.0 \
      --scheduler cosine --grad-clip 1.0 --num-workers 4 \
      --lambda-neg 1.0 --processing-order row \
      --seed <SEED> \
      --save_dir logs/warcraft_retrain_steps/<CONFIG>/run<SEED>

with, per config:

| config | flag | save_dir |
|---|---|---|
| lambda = 0 | `--lambda-l2-edit 0.0` | `logs/warcraft_retrain_steps/lambda0/run{0,1,2}` |
| lambda = 1 | `--lambda-l2-edit 1.0` | `logs/warcraft_retrain_steps/lambda1/run{0,1,2}` |

`beta_ce`, `gamma_focal`, `lambda_edge`, `lambda_pair`, `lambda_len` are all
left at their `0.0` argparse defaults — MinEdit only — and are recorded in each
run's `config.json` under `auxiliary_losses_off`. Seeds 0, 1, 2. No existing
checkpoint directory is written to.

The colleague's mean-norm variant of the lambda = 1 row is not implemented and
will not be; the Phase D report will state that beside the delta.

## Tripwire for Phase C

Seed 0 of lambda = 0 runs first and alone. Report before the other five launch:
wall-clock, loss-curve sanity against the original runs' shapes, and a 50-board
smoke through the full steps inference pipeline with `certify` on — 50/50
certificates expected regardless of accuracy.
