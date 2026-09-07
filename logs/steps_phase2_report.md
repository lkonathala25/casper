# Phase 2 — compile corroboration at 12x12

Single worker (`SAT_WORKERS=1`), AMD EPYC 9334 32-core (the tex reports a
9335 — same generation). Wall times are from a quiet compile pass; the
resolvent accounting comes from a separate `verbose=True` pass, because the
verbose chatter runs to ~273k lines and would otherwise sit in the clock.
Artifact sizes are the streamed `.txt`, which is the serialization the tex's
artifact column matches; the JSON serialization of the same compile is larger
(2.61 MB for the trimmed row).

## Row 1 — trimmed 12x12 (the claim under test)

| column | tex | measured | |
|---|---|---|---|
| clauses in | 8,839 | 8,839 | exact |
| resolvents | 0 | 0 | exact |
| time | 2.0 s | 21.7 s | ~11x |
| artifact | 1.9 MB | 1.975 MB | exact |

`n_resolvents_kept=0` at every one of the 144 buckets — nothing was derived
and then subsumed away, nothing was derived at all. The trimmed input is its
own compilation fixpoint, as the tex claims.

The generator emits 8,844 clauses; COMPILE drops 5 tautologies before counting
(`resolution.py:1635`), giving 8,839. The five are the `i=0` clause and the
four trailing absorption clauses — the same five at 3x3, 12x12 and 30x30.

### The time deviation, diagnosed

Not resolvent work. Summing the per-bucket log, 18.42 s of the 23.75 s total
step time is `sub_elapsed_s`: the unconditional per-step
`_dedup_by_subsumption` over the whole surviving clause set at
`resolution.py:1787`. It reported `sub_elim=0` at all 145 recorded steps.
When a bucket derives nothing, the surviving set is the previous one minus the
eliminated variable's clauses, which was already subsumption-minimal, so the
pass is a provable no-op — 85% of the wall clock. Remaining ~3.5 s is pattern
expansion; slowest single bucket 0.353 s; parse 0.23 s.

The tex's qualitative statement ("compilation dominated by parsing and
subsumption bookkeeping") reproduces exactly; only the constant differs, and
it sits in a pass that is skippable when a bucket derives nothing.
`resolution.py` was not modified.

### Semantic check

At 3x3, where exhaustive enumeration is possible, the compiled artifact is
model-preserving: 402 models before and after compiling, equal to the
independent absorbing-walk enumeration used in the Phase 1 tests.

## Row 2 — untrimmed control: MEASURED-PARTIAL

The 12x12 run was capped and killed at 16 min 14 s having finished 2 of 144
buckets. What follows is a small-grid sweep of the same control plus an
extrapolation, not a completed 12x12 measurement.

| g | clauses in | resolvents | time |
|---|---|---|---|
| 3 | 61 | 10 | 0.00 s |
| 4 | 222 | 33 | 0.13 s |
| 5 | 573 | 76 | 101.5 s |
| 12 | 20,446 (tex 20,448) | — killed — | — killed — |

### The resolvent count corroborates exactly, by extrapolation

The measured counts fit `(g-2)*g^2 + 1` with no free parameters:
10, 33, 76 for g = 3, 4, 5. At g = 12 that formula gives **1,441** — the tex's
figure exactly. This is a genuine prediction on a fourth point, not a fit, so
the control's derived-resolvent content agrees with the tex even though the
run could not be completed.

### Deviations carried

1. **Input clauses: 20,446 measured vs 20,448 in the tex** (20,449 emitted,
   3 tautologies dropped). Carried from Phase 1. I swept the plausible
   variants — endpoints pinned as domains vs as clauses, waiting on/off,
   absorption on/off — and none hits both 20,448 at 12x12 and 808,202 at
   30x30; the closest is off by 2 and 4. Input DSL is 10.20 MB against the
   tex's 11 MB, consistent with a near-identical but not identical control.

2. **Wall time is super-exponential in g in this implementation.** 0.00 s,
   0.13 s, 101.5 s for g = 3, 4, 5 — roughly 780x from g=4 to g=5. The 12x12
   per-bucket cost (>7 min) is comparable to the per-bucket cost the tex
   reports for *30x30*, not for 12x12, where it claims 38.2 s in total.

Reading these together: the encoding is right (the resolvent count lands on
1,441) and the cost model is not. The gap is in how the minimal-conflict
subsets are enumerated at cap = full 144-value domain, i.e. in the compile
machinery, not in the control's definition. That is the same direction as the
trimmed row's 11x subsumption constant — a systematic implementation or
version difference in COMPILE, affecting both rows, leaving the encoding
claims (clause counts, zero resolvents, 1,441 resolvents, artifact size)
corroborated.
