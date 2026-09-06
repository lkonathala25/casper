# Task: Reimplement the time-unrolled ("steps") encoding from steps_based_encoding.tex
# — with two inference improvements — and corroborate all reported findings at 12x12.

Work in phases, halt + report at each phase boundary. Nothing in the
edge-based lazy-exact pipeline changes; this is a parallel pipeline built
alongside it. The tex document is the spec; its worked examples are the
unit tests.

## Phase 1 — Encoding generator (tex: "Variables and transitions",
## "From the relation to signed clauses", "Domain trimming")

Implement `build_steps_encoding(g, trimmed=True)` producing the DSL file:

1. Variables X_0..X_L, L = g^2 - 1. Cell numbering row-major, s=0,
   t=g^2-1.
2. Trimmed domains D_i exactly per the tex:
   - forward:  d(s,v) <= i
   - parity:   i == d(s,v) mod 2
   - backward: d(v,t) <= L - i
   - t enters D_i only for i >= d(s,t); D_0={s}, D_L={t}.
   d(.,.) is Manhattan distance on the grid.
3. Clauses: for each i and each u in D_i, the implication
   X_i=u => X_{i+1} in succ(u) cap D_{i+1}, written as the signed clause
   [D_i \ {u}]:X_i  OR  [succ(u) cap D_{i+1}]:X_{i+1},
   where succ(u) = grid neighbours of u for u != t, and {t} for u = t
   (absorption). Drop the guard when D_i is a singleton (unary clause).
4. NO AllDifferent clauses (allow_cycles) — simplicity handled by loop
   erasure downstream, exactly as the tex prescribes.

UNIT TESTS (from the tex, must match exactly):
- 2x2: domains D_0={0}, D_1={1,2}, D_2={3}, D_3={3}; exactly the four
  clauses printed in the tex; exactly two models (0,1,3,3) and (0,2,3,3).
- 3x3: the 29-value domain table as printed; the three sample models all
  satisfy; the four listed exclusions all violate.

## Phase 2 — Compile corroboration at 12x12

Run the existing COMPILE on the generated artifacts:
- trimmed 12x12: EXPECT 8,839 clauses in, 0 resolvents, ~2 s, ~1.9 MB.
- untrimmed control (full domains, waiting variant per the tex table):
  EXPECT 20,448 clauses, 1,441 resolvents, ~38 s. Run this too — the
  contrast (trimming pre-derives what elimination would compute) is part
  of what we corroborate.
Report measured clauses / resolvents / wall time / artifact size against
the tex table. Any deviation is a finding, not a failure — report it.
(30x30 compile deferred; not needed for this phase.)

## Phase 3 — Inference with the two improvements

Sequential projection over X_0..X_L with:

IMPROVEMENT 1 (argmax selection — per the Statement):
  at each step, the feasible set is succ(x_{i-1}) cap D_i (at most 4
  cells + absorption). Select t* = argmax of the bridged scores over the
  feasible set. Do NOT compute per-candidate L2 distances — the Statement
  proves argmax equals the min-edit choice in the eps->0+ limit.
  A/B gate: on 200 boards, compare argmax selection vs full per-candidate
  distance computation at the implementation's actual eps — report the
  divergence rate (expected ~0). If nonzero, report examples; do not
  silently adopt.

IMPROVEMENT 2 (early cut at absorption):
  the moment x_i = t, STOP the walk — all remaining X_j are t by
  absorption; fill them without projection. This changes no output (the
  encoding forces the padding) and cuts the horizon from L to the actual
  arrival time. Assert equivalence on 50 boards vs the un-cut version.

Then: loop erasure (walk the sequence; on revisiting a cell, snip the
detour), decode the erased sequence to an edge set, and run the SAME
independent validity check as the lazy-exact pipeline (endpoints degree
1, others {0,2}, single component). Every returned instance must pass —
same certificate discipline as before. By construction the walk cannot
dead-end (backward trimming guarantees a legal successor exists at every
reachable state) — assert this too (counter, expected 0).

## Phase 4 — The bridge: plain broadcast (lambda=0 candidate)

Implement the simplest bridge only: the tick-score of cell v at time i =
the network's on-path score for cell v, restricted to D_i. No weighting,
no normalization. This is our reading of the reference table's lambda=0
row; the lambda=1 mean-norm variant is PARKED — do not implement, do not
guess. The report must state explicitly: "bridge = plain broadcast,
assumed equivalent to the reference lambda=0 — unverified assumption."

## Phase 5 — Evaluation and corroboration at 12x12

Same checkpoints, seeds, undoctored GT, patched harness, census slices
as the lazy-exact evaluation. Corroboration target is the lambda=0 row
ONLY:

  config      n     optimal%      exact
  lambda=0    6/6   68.3 +/- 2.9  0.574 +/- 0.042

Report our measured numbers against it, plus the standard columns (EM,
Optimal%, Hamming, Consistent% with certificate failures, ratio,
ms/board) and the comparison rows: monotone-as-published, lazy-exact,
steps-encoding. Context for reading: the reference numbers (57.4% exact,
68.3% optimal) sit BELOW lazy-exact (70.7 EM / 73.6 Optimal) — landing
near the reference is corroboration; landing far from it means diagnose
in this order: (1) phases 1-3 bugs (should be excluded by the tex
worked-example tests and compile corroboration), (2) the broadcast
reading of lambda=0 — take the concrete number to the colleague before
guessing further. Also report loop-erasure statistics: % of boards
needing any erasure, mean cells snipped — the analogue of the ring rate.

## Ground rules
Seeds + logs as always; results JSON to logs/; edge-based pipeline,
resolution.py, Sudoku untouched; halt at each phase boundary.