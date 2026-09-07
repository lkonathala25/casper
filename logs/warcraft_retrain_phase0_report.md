# Phase 0 — recon + budget for the full-alphabet 12x12 retrain

No training run. Three blocking items surfaced; all need a decision before
Phase 1 builds anything.

## 1. Original recipe and cost

From `run_warcraft_12x12.sh` and `src_warcraft/train.py` defaults:

| knob | value |
|---|---|
| epochs / patience | 300 / 30 (early stopping) |
| optimizer | Adam, lr 5e-4, weight decay 0.0 |
| scheduler | cosine, `eta_min = lr * 0.01` |
| batch size | 128 |
| grad clip | 1.0 |
| base loss | `WarcraftMaskedLoss` (per-node MinEdit), `lambda_neg = 1.0` |
| aux (node) | `beta_ce = 0.1`, `gamma_focal = 2.0` |
| aux (edge) | `lambda_edge = 0.5`, `lambda_pair = 0.3`, `lambda_len = 0.05` |
| self-distillation | `lambda_l2_edit = 1.0` |
| processing order | row |

Measured wall-clock, `warcraft_ablation_edge_aux_v2/crl_in_training`
(`epoch_time_s` column of the 3-run CSV):

| run | epochs | mean epoch | total |
|---|---|---|---|
| 0 | 105 | 15.3 s | 0.45 h |
| 1 | 91 | 14.0 s | 0.35 h |
| 2 | 140 | 15.6 s | 0.61 h |

3 seeds = 1.41 h. Early stopping fires around 90–140 epochs, well short of the
300 cap.

**Projected cost for this retrain.** Measured directly: the lazy-exact layer
under the full alphabet costs a median 0.104 s per 128-board batch (5 repeats,
0.095–0.106 s), i.e. +8.2 s/epoch over 79 train batches. Removing the edge-aux
losses claws some of that back.

| config | est. epoch | est. per run | 3 seeds |
|---|---|---|---|
| lambda = 0 (no train-time layer) | 12–16 s | ~0.44 h | ~1.3 h |
| lambda = 1 (single-pass layer per batch) | 20–24 s | ~0.68 h | ~2.1 h |

**Total ~3.4 h sequential** on one GPU, assuming epoch counts like the
original's. Caveat: all three RTX 4500 Ada GPUs are currently at 67–100%
utilisation with other jobs, so wall-clock depends on what else is running;
spreading the 3 seeds across GPUs would bring it to ~1.2–1.5 h.

## 2. The `--alphabet full` path — verified end-to-end

- **Head dim.** `build_node_categories(12, alphabet='full')` gives
  `total_dim = 868` (monotone gives 628), and `train.py` builds the model from
  that `var_ranges`, so the head is 868-dim. Confirmed.
- **Labels.** `WarcraftDataset` converts `{split}_edge_labels.npy` to node
  labels on the fly through `edge_labels_to_node_labels(node_info, ...)`, so
  passing `--alphabet full` re-derives the labels under the full alphabet. No
  separate label file is involved.
- **Fail-loud conversion, re-asserted just now over the whole dataset:**

  | alphabet | train (10,000) | val (1,000) | test (1,000) |
  |---|---|---|---|
  | monotone | **raises** `UnrepresentablePatternError` | raises | raises |
  | full | **clean** | clean | clean |

  Zero unrepresentable patterns under `full`, exactly as the ws1 report had it,
  and the monotone path still raises rather than silently relabelling — which
  is the corruption this retrain removes.

## 3. The self-distillation term — where it hooks, and what breaks

**Where.** `train_epoch_masked` (`train.py:224-226`):

    l2_edit = (probs - projected.detach()).pow(2).sum(dim=1).mean()
    loss = loss + lambda_l2_edit * l2_edit

with `projected` produced in `ResNet18NodeCategorical.forward` (`model.py:57`):

    projected, info = self.constraint_layer(probs.detach())

`apply_constraints_in_train` is `lambda_l2_edit > 0.0`, so at lambda = 0 the
layer never runs during training.

**Gradient treatment used by the original: stopped-gradient distillation
target, enforced twice** — the layer is fed `probs.detach()`, and the loss uses
`projected.detach()`. Gradient flows only through `probs`. This is cleanly
implementable under the full alphabet — nothing about the stop-gradient depends
on the alphabet — so I will match it exactly rather than re-decide.

### FLAG — the layer is worse than incompatible, it is silently wrong

`train.py` constructs the model with **no** `constraint_mode`, `alphabet` or
`artifact` argument (`train.py:423-426`), so it always gets the default
monotone `WarcraftConstraintLayer`. That layer calls
`build_node_categories(grid_size)` with no alphabet (`constraint_layer.py:16`),
so its internal `total_dim` is 628.

Fed an 868-dim input it **returns a (batch, 868) tensor and raises nothing** —
I ran it. The per-node offsets are simply wrong, so the distillation target
would be garbage and no error would ever surface.

**This is not confined to the lambda = 1 term.** Under `crl_in_training`,
model selection is (`train.py:538-541`):

    select_metric = 0.5 * val["raw_exact"] + 0.5 * val["proj_exact"]

`proj_exact` comes from the same layer. So under `--alphabet full` today, both
configs would select checkpoints on a corrupted metric — lambda = 0 included,
even though it never runs the layer during training.

### Your proposal, confirmed implementable

Lazy-exact degree projection with the ring-repair loop disabled, as the
training-time target: `WarcraftLazyExactLayer(grid_size=12, alphabet='full',
max_rounds=1)`. Verified:

- `max_rounds=1` runs `_project_one` exactly once with no repair iteration.
- On the probe batches, `rounds_mean = 1.0` and `boards_needing_refinement = 0`
  even at `max_rounds=20` — rings are rare, as you expected.
- Cost 0.104 s per 128-batch (above). The full loop with certificates stays
  inference-only.

### Two decisions I need before Phase 1

**(a) Which layer does eval / model selection use?** The distillation target is
settled, but `proj_exact` still needs a working layer. Options:

1. **Recommended** — plumb `constraint_mode='lazy_exact', alphabet='full'` into
   the model and use `crl_in_training` for *both* configs, so the selection
   metric is identical across lambda and the lambda delta stays clean. The
   eval-time layer may keep its full ring-repair loop; only the training-time
   term is single-pass.
2. Use `crl_in_training_eval_off` for lambda = 0 so selection is raw-only. This
   makes selection differ between the two configs and confounds the delta —
   I would not.

Either way this needs a small **additive** change to `src_warcraft/train.py`
(new `--constraint-mode` / `--max-rounds` flags, defaults preserving today's
behaviour) plus passing `alphabet` through to the model. `train.py` is shared
with the edge pipeline, so I want that change acknowledged rather than assumed.

**(b) Does "no edge-aux" also drop the node-level CE?** The brief says both
configs carry "NO edge-aux losses" and that lambda = 0 uses "the standard
per-node MinEdit loss ONLY", but also "everything else identical to the
original recipe". Those conflict for `beta_ce = 0.1` / `gamma_focal = 2.0`,
which are node-level auxiliaries, not edge ones. `lambda_edge`, `lambda_pair`,
`lambda_len` are unambiguously out.

My reading is **MinEdit only** — drop `beta_ce` and `gamma_focal` too, since
that is the cleaner mirror of a pure lambda ablation and matches "ONLY"
literally. But it materially changes the backbone, so I am not choosing it
silently.

## Planned invocation (pending the two decisions)

    python src_warcraft/train.py \
      --data_dir <ws1 12x12 preprocessed_4conn> \
      --save_dir logs/warcraft_retrain_full/{lambda0,lambda1}/run{0,1,2} \
      --grid-size 12 --alphabet full --condition crl_in_training --n-runs 1 \
      --epochs 300 --batch-size 128 --lr 5e-4 --patience 30 \
      --scheduler cosine --grad-clip 1.0 --num-workers 4 \
      --lambda-neg 1.0 --processing-order row \
      --lambda-l2-edit {0.0, 1.0} \
      --constraint-mode lazy_exact            # new flag, decision (a)
      # beta-ce / gamma-focal / lambda-edge / pair / len: decision (b)

The lambda = 1 config is the colleague's "with layer" row. **The mean-norm
variant of their lambda = 1 row is not implemented and will not be — it is
undefined for us**, and the Phase 3 report will say so beside the delta
comparison.
