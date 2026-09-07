# Phase 3 — guided walk on the trimmed 12x12 artifact

Synthetic scores throughout; the real bridge is Phase 4. Seed 0, eps 1e-3
(the surrogate threshold the edge-based pipeline uses), 200 boards per score
regime, 50 of them also run un-cut. Results in `steps_phase3_results.json`
(float32, the pipeline's real dtype) and `steps_phase3_results_fp64.json`.
Run under the `spl` conda env; tests pass under `dpl`.

Three score regimes, because the mechanism behaves differently under each:
`smooth` (iid gaussian — an unguided walk that uses the whole horizon),
`guided` (tilted toward the target, the realistic proxy for a trained
network), `ties` (hard-quantised, manufacturing exact ties among feasible
candidates — the adversarial case for Improvement 1).

## The feasible set is the artifact's

`feasible_matches_artifact = True`: for every step i and every u in D_i, the
walk's succ(u) cap D_{i+1} equals what the compiled artifact's own clauses
license. This is checked, not assumed — it is the reason the walk may use the
cheap successor computation instead of a generic clause-feasibility query, and
it holds because the trimmed compile derived zero resolvents.

## Gates

| gate | result |
|---|---|
| dead ends | **0** across all 600 walks (58,983 decision steps) |
| Improvement 1, strict divergences | **0** in every regime and both precisions |
| Improvement 2, early-cut mismatches | **0** on 150 boards (50 per regime) |
| validity certificates | **600 / 600**, 0 failures |

### Improvement 1 — argmax vs full per-candidate min-edit distance

| regime | decisions | divergences fp32 | fp64 | strict |
|---|---|---|---|---|
| smooth | 27,113 | 1 | 0 | 0 |
| guided | 4,488 | 2 | 0 | 0 |
| ties | 27,382 | 384 | 373 | 0 |

A divergence is counted whenever argmin of the per-candidate MinEditL2
distance picks a different cell than argmax of the score. It is *strict* only
if the argmax candidate was both strictly higher-scoring and strictly
higher-distance — i.e. min-edit genuinely preferred something argmax rejected.
There were none.

Every observed divergence has the same signature: the two candidates'
min-edit distances are equal to within float32 resolution (largest observed
gap 7.5e-9), so argmin falls to index order. Recomputing the whole sweep in
float64 removes them in the smooth and guided regimes entirely. The 373 that
survive in `ties` are exact probability ties by construction — both candidates
are equally the argmax, so neither rule has a preference to express.

Conclusion: argmax selection reproduces the min-edit choice at the
implementation's real eps, with no case where the two rules genuinely
disagreed. Adopting it is safe on this evidence.

### Improvement 2 — early cut at absorption

Zero mismatches: the cut sequence equals the un-cut sequence on all 150
boards checked. It also earns its keep only when the walk arrives early —
mean arrival is step 139.8 of 143 under `smooth` but 22.4 under `guided`,
where the cut removes ~85% of the horizon and per-board time drops from
2.88 ms to 0.42 ms.

### Loop erasure

| regime | boards needing erasure | mean cells snipped (all) | mean when erased |
|---|---|---|---|
| smooth | 100.0% | 112.24 | 112.24 |
| guided | 11.5% | 0.30 | 2.61 |
| ties | 100.0% | 117.61 | 117.61 |

The `guided` row is the one to compare against the lazy-exact ring rate; the
other two describe a walk driven by noise and are included to show the
erasure machinery handles the pathological case. These numbers will move once
the real bridge lands, so treat them as a mechanism check rather than the
reported statistic.

## Certificate discipline

Decoded edge sets are checked by `_is_valid_src_sink_path` imported directly
from `src_warcraft/compute_cost_ratio.py` — the same function the edge-based
pipeline uses, not a reimplementation: endpoints degree 1, every other
involved vertex degree 2, connected, #edges = #verts - 1. All 600 returned
instances pass. The test suite also confirms the check is a real gate: the
tex's cyclic 3x3 model fails it before erasure and passes after.

## Caveat

Scores are synthetic. Everything above establishes that the mechanism is
correct and that the two improvements are safe; none of it says anything
about accuracy. That waits on Phase 4's bridge.
