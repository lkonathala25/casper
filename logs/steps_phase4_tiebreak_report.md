# Phase 4b — adopted tie-break, re-run gates

Seed 0, 12x12, eps 1e-3, inference-only. Results in
`steps_phase4_bridge.json`, `steps_phase3_results.json` (float32, the
pipeline's real dtype) and `steps_phase3_results_fp64.json`.

## 1. The adopted rule

`StepsWalkDecoder._break_tie`, now the default (`tie_break="no-backtrack"`):

> Ties are broken against immediate reversal because a reversal is a
> provably-erased move — loop erasure deletes it downstream by construction —
> so among equally-scored candidates we prefer one that survives to the
> output. Non-tie selections are untouched; the Statement's argmax result
> governs those and is unaffected.

It fires only on an exact tie at the top score, and only to prefer a
non-reversal among the tied candidates. Any strict winner is taken regardless
of direction. The same rule is applied to the min-edit branch (ties at minimum
distance), so the A/B compares two rules that handle ties identically. Three
unit tests pin the semantics, including that a strictly-winning reversal is
still taken.

## 2. Re-run gates

### GT pass-through — passes

| | before (index) | **adopted (no-backtrack)** |
|---|---|---|
| exact match | 94.3% | **100.0%** |
| certificates | 1000/1000 | **1000/1000** |
| dead ends | 0 | **0** |
| erasure rate | 100.0% | **0.0%** |
| mean cells snipped | 120.0 | **0.00** |
| mean arrival | 142.0 | **22.17** |
| ms/board | 2.63 | **0.44** |

### Phase 3 argmax-vs-min-edit A/B, re-run with the tie-break active — passes

| dtype | regime | decisions | divergences | **strict** |
|---|---|---|---|---|
| float32 | smooth | 27,113 | 2 | **0** |
| float32 | guided | 4,488 | 2 | **0** |
| float32 | ties | 27,187 | 436 | **0** |
| float64 | smooth | 27,113 | 0 | **0** |
| float64 | guided | 4,488 | 0 | **0** |
| float64 | ties | 27,187 | 379 | **0** |

Also unchanged: 0 dead ends, 0 early-cut mismatches, 600/600 certificates.

### 50-board checkpoint smoke — **unchanged, and this is the finding**

| | before (index) | adopted (no-backtrack) |
|---|---|---|
| exact match | 78.0% | 78.0% |
| certificates | 50/50 | 50/50 |
| dead ends | 0 | 0 |
| erasure rate | 100.0% | 100.0% |
| mean cells snipped | 120.0 | 120.0 |
| mean arrival | 142.0 | 142.0 |
| ms/board | 2.63 | 2.64 |

The adopted rule does not move the real-checkpoint numbers at all, because on
real network scores it never fires. Measured over the 6,871 multi-candidate
decision steps on those 50 boards:

- **exact ties at the top score: 0 (0.0%)**
- **the reversal cell is the sole strict top scorer: 6,000 (87.3%)**

Against the same measurement on ground-truth scores (1,106 decision steps):
95.5% exact ties at the top, 0% strict-winning reversals. That is the whole
difference between the two gates.

So the stall has two distinct causes, and the adopted rule addresses only one:

- Under **ground-truth** scores the reversal ties with the forward move
  (both exactly 1.0). The tie-break resolves it, and pass-through is exact.
- Under **real** scores the reversal is a *strict* win 87% of the time — the
  cell already visited carries higher P(on-path) than the cell ahead. No
  tie-break can touch that, by design: "any strict winner is taken regardless
  of direction."

**The 78% -> 80% figure in the Phase 4 report is therefore not reproducible
under the adopted rule.** It came from the broader diagnostic — dropping the
previous cell from the candidate set whenever anything else was available,
tie or not — which the adopted rule deliberately narrowed. That was my
diagnostic's semantics, not the approved rule's, and the difference is
material: the narrow rule leaves the real-checkpoint machinery baseline
exactly where it was.

Consequence for Phase 5: the machinery baseline going in is still
stall-dominated — 100% erasure, 120.0 cells snipped, arrival 142.0 of 143,
2.64 ms/board. The intended replacement did not happen.

## 3. Open question for the colleague

Whether the reference lambda=0 runs handled ties this way, or whether
lambda>0 introduces the time-dependence that avoids the stall.

Our numbers to attach:

- GT pass-through, index tie-break -> adopted tie-break: **94.3% -> 100.0%**
  exact match; erasure 100% -> 0%; arrival 142.0 -> 22.2.
- Real checkpoint (`best_run0`, 50 boards), same change: **78% -> 78%**
  (no movement). The earlier 78% -> 80% came from the broader
  drop-the-previous-cell diagnostic, not from the adopted tie-break.
- Why: on real scores **0.0%** of decision steps have an exact tie at the top,
  and the reversal cell is the **sole strict top scorer at 87.3%** of them.
  On GT scores it is 95.5% ties and 0% strict-winning reversals.

The sharper form of the question this suggests: a time-invariant broadcast
score gives the walk no forward pressure at all — the cell just left scores
higher than the cell ahead, strictly, most of the time. If the reference
lambda=0 row really is plain broadcast, something else in their pipeline must
supply that pressure; if it does not, the lambda=1 mean-norm variant (parked,
per the Phase 4 brief) may be doing exactly that, in which case lambda=0 is
not the row to corroborate against with this bridge.

## Files

`src_steps/steps_walk.py` (tie-break + docstring), `examples/eval_steps_walk.py`
and `examples/eval_steps_bridge.py` (`--tie-break`, before/after rows),
`tests/test_steps_walk.py` (+4 tests). 75 tests pass across the three steps
suites. Edge pipeline, `resolution.py` and Sudoku untouched.
