# Phase A — the reverse bridge and the steps projection layer

New component: `src_steps/steps_layer.py`. No training run; no existing file
modified. Gate data in `steps_layer_phaseA_gates.json`.

## The construction chosen

`StepsProjectionLayer` is distribution-in / distribution-out, same shape, so
the L2 term and `proj_exact` are well defined against it. The chain:

    per-node categorical probs
      -> forward bridge: P(on-path) = 1 - P(skip) per cell, then per-tick
         scores (time-aware, BETA = 1.0)
      -> guided walk over the compiled trimmed domains, feasible set at tick i
         is succ(x_{i-1}) cap D_i, adopted tie-break + reversal ban
      -> loop erasure  ->  edge set
      -> REVERSE BRIDGE  ->  per-node target category
      -> per-node min-edit onto that target
      -> corrected per-node categorical distribution

**Reverse bridge, exactly.** The erased walk is a simple s-t path, so every
node's incident on-edge pattern is determined, and under the full alphabet
exactly one category matches it (nodes off the path match `skip`). The target
is recovered with `CategoricalEdgeBridge.edges_to_categories` — the same
fail-loud converter the edge pipeline already uses, not a reimplementation.

**The emitted distribution is not one-hot.** Each node's original distribution
is minimally edited so its argmax becomes the target, via the same
`vectorised_positive_head` the lazy-exact layer applies. Mass is conserved per
node block and categories outside the active set keep their probability, so
the layer stays a minimal correction rather than a hard relabel — which is
what makes `||probs - projected.detach()||^2` the same kind of quantity it was
in the original recipe.

There is **no "soft where mass is spread" branch**: the walk commits to one
cell per tick, so the target category is always unique. The softness lives
entirely in the min-edit, exactly as it does in the edge pipeline. Stating
this explicitly because the brief left it conditional.

**On "ring-repair OFF, max_rounds=1":** there is no ring-repair loop in this
layer to disable. The steps encoding admits cyclic walks by construction and
loop erasure removes them in a single deterministic pass — that is the
structural replacement for the lazy-exact ring loop, not a setting. What the
training-time default does turn off is `certify`, the independent validity
check, which is the inference-only dressing. Flagging the wording so it does
not read later as a knob that was silently ignored.

## Gates — all pass

| gate | result |
|---|---|
| 1. GT round trip, 1,000 boards | **1000 / 1000** exact category match (100.0%) |
| 2. Shape / consistency | `(16, 868)`; nonnegative (min 0.064); per-node sums 1.0, max deviation 2.4e-7 |
| 3. Determinism | identical, max abs diff **0.0** |
| 4. Cost, 128-board batch, single pass, no certificate | **73.9 ms** median (73.7–76.4) on synthetic probs; **76.1 ms** on real network probs |

Gate 1 is the load-bearing one: GT node labels -> one-hot -> forward bridge ->
projection -> reverse bridge -> per-node argmax reproduces the GT node labels
on every test board, with zero mismatches. Perfect information survives the
full round trip.

Gate 4 was measured twice — once on synthetic distributions and once on real
per-node probabilities from an existing checkpoint widened to the full
alphabet — because the walk's cost depends on how peaked the scores are. The
two agree to within 3%, so the budget number is not an artefact of synthetic
input: **~6.0 s/epoch** over 79 train batches.

Also checked, rather than assumed: the reverse bridge's fail-loud guard is
live — feeding it an all-edges-on vector raises `UnrepresentablePatternError`.

## Cost implication for Phase B/C

~6.0 s/epoch added at lambda = 1, against the ~8.2 s/epoch measured for the
lazy-exact layer in Phase 0, so the steps layer is slightly cheaper. Note the
selection metric needs `proj_exact` in **both** configs, so lambda = 0 also
pays an eval-time cost — 1,000 val boards is 8 batches, about 0.6 s/epoch,
negligible.

Revised estimate against the Phase 0 baseline (~15 s/epoch, 90–140 epochs):

| config | est. epoch | per run | 3 seeds |
|---|---|---|---|
| lambda = 0 | ~12–16 s | ~0.44 h | ~1.3 h |
| lambda = 1 | ~18–22 s | ~0.62 h | ~1.9 h |

~3.2 h sequential, subject to GPU contention as noted in Phase 0.

## Tests

`tests/test_steps_layer.py`, 8 tests: round trip, target-vs-decoded-edge-set
agreement, output-is-not-one-hot, per-block mass conservation, untouched
categories preserved, determinism, no dead ends, and the certify path. 83
tests pass across the four steps suites.

`src_warcraft/` and `src/resolution.py` are unmodified — `git status` matches
the session-start snapshot exactly. The additive `train.py` changes are
Phase B and have not been made.
