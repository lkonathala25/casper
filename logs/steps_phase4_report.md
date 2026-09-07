# Phase 4 — the plain-broadcast bridge

> **Superseded in part by `steps_phase4_tiebreak_report.md`.** The
> no-backtrack tie-break was adopted in a narrower form than the diagnostic
> run here — exact ties only — so the 78% -> 80% real-checkpoint figure below
> does not reproduce under the adopted rule. See that report.

**bridge = plain broadcast, assumed equivalent to the reference lambda=0 —
unverified assumption.**

the tex specifies no training regime; inference-only is our choice, matching
the lazy-exact evaluation.

Seed 0, grid 12x12, checkpoint `best_run0.pt` from
`warcraft_ablation_edge_aux_v2/crl_in_training`, test split of 1,000 boards
from `warcraft_crl/warcraft_shortest_path_oneskin/12x12/preprocessed_4conn`.
Results in `steps_phase4_bridge.json`. The checkpoint carries a monotone
(628-dim) head and is widened by `embed_monotone_head_into_full`, the same
path the lazy-exact evaluation takes, so both pipelines read identical
network outputs.

## The definitional choice, made loudly

The head is per-node categorical, not a scalar per cell. Interior nodes carry
a `skip` category (all incident edges off) plus one category per unordered
pair of incident edges — 7 at a degree-4 node under the full alphabet.

**We define the on-path score of cell v as P(on-path) = 1 - P(skip).**

The source and sink have no `skip` category at all — every category there
lights exactly one incident edge — so their score is identically 1.0, which
agrees with the encoding pinning D_0 = {s} and D_L = {t}.

Alternatives considered and not taken:

- **max over the non-skip categories.** Reads the single most likely routing
  pattern rather than total on-path mass. Discards the spread across
  directions, which is information the walk can use.
- **derive per-cell scores from the edge scores of `CategoricalEdgeBridge`
  (mean or max over incident edges).** Routes the scalar through the edge
  representation the steps encoding is meant to avoid, and double-counts:
  each edge score is already an average of its two endpoints.
- **the raw logit of the skip category.** Not comparable across nodes with
  different category counts.

`1 - P(skip)` is the only one of these that is exactly the network's belief
that the path touches the cell, which is what "on-path score" names.

No time weighting, no normalization: the same scalar is broadcast to every
tick and simply restricted to D_i. Verified rather than asserted — the walk
softmaxes over D_i only to hand MinEditL2 a distribution, and
`raw_argmax_equals_softmax_argmax = true` over every decision step on 20
boards confirms that step changes no selection.

## Gates

| gate | result |
|---|---|
| GT pass-through, certificates | **1000 / 1000** |
| GT pass-through, dead ends | **0** |
| GT pass-through, exact match | **94.3%** — not the expected 100 |
| checkpoint smoke, certificates | **50 / 50** |
| checkpoint smoke, dead ends | **0** |
| determinism | identical on repeat |

## The pass-through gate did not come out at 100, and why

94.3% exact match, 57 boards missed. Every one still passed the certificate,
so these are valid s-t paths that are not the ground-truth path.

First hypothesis — chorded GT paths, where the vertex set alone admits a
shortcut — was **wrong**: zero of the 1,000 GT paths have a chord.

The real cause is visible in the walk traces. A time-invariant per-cell score
gives the cell you came from and the cell you are going to the **same score**.
At every step the feasible set contains both, they tie at 1.0, and the tie
breaks on lowest cell index — so the walk steps backwards. It then oscillates
along the ground-truth path for the whole horizon until the backward-trimming
cone forces arrival (mean arrival 142 of 143), and the cone-forced tail runs
through off-path cells. Loop erasure recovers the ground-truth prefix on 943
boards; on 57 the cone-driven tail differs from the ground-truth tail.

Isolating diagnostic, **run but not adopted**: changing only the tie-break, so
that the immediately previous cell is dropped from the candidate set whenever
anything else is available, gives

| tie-break | exact match | erasure rate | cells snipped | arrival | ms/board |
|---|---|---|---|---|---|
| index (as built) | 94.3% | 100.0% | 120.0 | 142.0 | 2.65 |
| no-backtrack (diagnostic) | **100.0%** | 0.0% | 0.00 | 22.2 | 0.41 |

So the bridge does preserve perfect information end-to-end; the 5.7% is
entirely the stall, and the stall is a property of a time-invariant score
under an argmax rule, not of the `1 - P(skip)` scalar. On the real checkpoint
the same change moves exact match 78.0% -> 80.0% and erasure 100% -> 2%.

I am not adopting the no-backtrack tie-break — it is a change to the selection
rule, not to the bridge, and the Statement's argmax result says nothing about
tie handling. Taking the concrete numbers above to the colleague before
guessing further is the next step, and it should happen before Phase 5, since
the reported loop-erasure statistics and the ms/board differ by 6x between the
two rows.

## Checkpoint smoke — no accuracy claims

50 boards, `best_run0`: 50/50 certificates, 0 dead ends, 100% of boards need
erasure with 120.0 cells snipped on average, arrival 142.0, 2.65 ms/board.
Three example walks are recorded in the results JSON; board 0 erases to
`[0, 12, 24, 36, 48, 60, 72, 84, 85, 97, 98, 110, ..., 119, 131, 143]`, a
legal staircase from corner to corner, which is the eyeball sanity wanted.

The erasure rate and ms/board here are dominated by the stall described
above, not by network quality. They are not the numbers to carry into
Phase 5 until the tie-break question is settled.

## Files

`src_steps/steps_bridge.py`, `examples/eval_steps_bridge.py`,
`tests/test_steps_bridge.py` (8 tests; 71 pass across the three steps suites).
Edge pipeline, `resolution.py` and Sudoku untouched.
