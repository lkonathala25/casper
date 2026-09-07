# Phase 5 Stage 1 — the 2x2 bridge-reading x decoder screen

bridge readings evaluated because the reference's λ=0 definition is unavailable

inference-only by our choice; the tex specifies no training regime.

Seed 0, `best_run0.pt` from `warcraft_ablation_edge_aux_v2/crl_in_training`,
full 1,000-board test set, undoctored GT. Results in
`steps_phase5_stage1.json`.

## Setup checks

**Checkpoint count.** 25 trained 12x12 checkpoints exist (628-dim monotone
heads; the other 24 `best_run*.pt` under `logs/warcraft*` are 30x30 at 4264).
They are ablation *arms*, not seeds of one configuration:

- `warcraft_ablation_edge_aux_v2/crl_in_training` — **3 seeds**, the config the
  lazy-exact evaluation uses and where `best_run0` lives.
- `warcraft_ablation_edge_aux_v2_diagonal/crl_in_training` — 3 more seeds,
  differing only in processing order, which `run_warcraft_12x12.sh` calls
  "empirically equivalent on this task".
- The remaining 19 are other arms (`no_crl`, `crl_inference_only`, `_v2`,
  `_v3`, `edge_aux` v1) or smoke runs.

So Stage 2 has **3 strictly comparable seeds**, or 6 if the diagonal-ordering
arm counts against the reference's 6. That is a call to make, not one I made.

**Time-aware component.** Built and verified.
`tick-score(v, i) = P(on-path)(v) * exp(-BETA * |d(s,v) - i|)`,
**BETA = 1.0**, fixed, not swept, documented in `TimeAwareBridge`. The trimmed
domains enforce `i = d(s,v) (mod 2)`, so the disagreement is always even and
the smallest possible one already costs `e^-2 = 0.135` — a reversal must score
~7.4x higher than the forward move to still be taken.

## Reference for reading

    config     n     optimal%      exact
    lambda=0   6/6   68.3 +/- 2.9  0.574 +/- 0.042

**Stage 1 is single-seed and can indicate neighbourhood only, not a match.**

## Stage 1 table

| cell | bridge | decoder | EM % | exact | Optimal % | Hamming | Consistent % | cert fail | ratio | ms/board | erasure % | snipped | arrival | strict-rev % | dead ends |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | plain broadcast | adopted tie-break only | 77.0 | 0.770 | 82.0 | 4.55 | 100.0 | 0 | 1.0308 | 2.72 | 100.0 | 120.00 | 142.0 | 87.3 | 0 |
| B | plain broadcast | no-backtrack | 76.0 | 0.760 | 81.2 | 4.70 | 100.0 | 0 | 1.0419 | 0.52 | 5.0 | 4.75 | 26.9 | 0.0 | 0 |
| C | time-aware | adopted tie-break only | 74.9 | 0.749 | 79.2 | 6.92 | 100.0 | 0 | 1.1395 | 0.72 | 13.4 | 13.92 | 36.1 | 5.6 | 0 |
| D | time-aware | no-backtrack | 76.9 | 0.769 | 81.9 | 4.44 | 100.0 | 0 | 1.0312 | 0.45 | 1.2 | 1.32 | 23.5 | 0.0 | 0 |

Horizon 143 throughout. Every cell: 1000/1000 certificates, zero certification
failures, zero dead ends.

## Anchor: the same checkpoint under the repo's own edge-based evaluation

`extra_metrics_12x12.json` for this exact checkpoint records **exact 76.4%,
optimal (`pct_eq_1.0`) 80.8%, ratio mean 1.0330, valid 100%**. Cells A and D
land within 0.6 and 1.2 points of that on both metrics, with a matching cost
ratio. The steps pipeline is measuring the same thing as the edge pipeline,
correctly.

## Read

No cell is in the reference's neighbourhood. All four sit far above it — exact
0.749–0.770 against 0.574 +/- 0.042, and optimal 79.2–82.0 against
68.3 +/- 2.9, roughly four sigma clear on both metrics in the same direction.
The anchor above says this is not our pipeline being wrong: our numbers
reproduce the repo's own edge-based evaluation of the identical checkpoint
(76.4 / 80.8), so the gap to the reference is a property of the checkpoints,
not of the bridge reading or the decoder. As for where the forward pressure
lives, the strict-reversal rate answers cleanly: 87.3% in A, 0% in B and D,
and 5.6% in C — so **both** the decoder ban and the time-aware bridge supply
genuine forward pressure, independently, and either one alone collapses the
stall. But neither is where accuracy lives. Across a 2.1-point EM spread the
machinery changes by an order of magnitude — erasure 100% -> 1.2%, cells
snipped 120.0 -> 1.3, arrival 142 -> 23.5, 2.72 -> 0.45 ms/board — which says
loop erasure was already rescuing the stalled walk almost perfectly, at a ~6x
runtime cost and near-zero accuracy cost. Accuracy here is set by the
network's per-cell on-path scores, not by walk dynamics. The one cell that
moves accuracy materially is C, and it moves it *down* (74.9 EM, ratio 1.1395,
Hamming 6.92): time-awareness at BETA = 1.0 without the reversal ban pulls the
walk onto the tick-matched diagonal even where the scores disagree, producing
longer and costlier paths.

## Consequence for Stage 2

Stage 2 as scoped cannot close this gap. The seed spread in the comparable
config is tiny — the recorded exact for run0/run1/run2 is 76.4 / 76.1 / 75.6,
a spread of 0.8 points against a reference exact standard deviation of 4.2
points — so adding seeds will tighten our error bar around ~76, not move it
toward 57.4. The gap is systematic, and the diagnosis order from the Phase 5
brief now points past step (2): phases 1–3 are excluded by the worked-example
tests and the compile corroboration, and step (2), the broadcast reading of
λ=0, is excluded by this screen, since all four readings — including one with
real forward pressure — land in the same place, ~4 sigma from the reference.
That points at the remaining possibility: the reference's λ=0 runs are not
these checkpoints. Worth taking to the colleague with the numbers above before
spending Stage 2.
