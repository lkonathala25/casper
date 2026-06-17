# Compile speedup — session summary

## Problem
- The 4×4 sudoku compile took **25h40m**, with 92% of the runtime concentrated in two variables.

## Work completed
1. **Profiling.** Added per-variable timing. One variable (y_4) accounted for 72% of total runtime, and roughly 72% of the clauses generated for it are tautologies that are immediately discarded.
2. **Early-pruning attempt — no effect.** Reordering the expansion to detect tautologies sooner had no measurable effect on sudoku: clause widths are uniform, so there is no useful signal to reorder by.
3. **Multi-processing — primary speedup.** The compile previously used a single core of 128. The per-pattern expansion is independent work, so it was distributed across worker processes (`SAT_WORKERS=N`). The 4×4 compile fell from 25h40m to ~4.2 hours (~6× overall; ~8× on y_4). The output was verified identical to the serial result.
4. **Parallel cleanup.** Once the expansion was parallelized, the final subsumption (redundancy-removal) pass became the dominant single-core cost, so it was parallelized as well.
5. **Incremental tautology stop.** Clause construction now halts the moment a clause becomes a tautology rather than completing it. Implemented; not yet validated.

## Observations from multi-processing
- **Raw resolvents increased.** Workers do not share their in-progress results, so redundant clauses that the serial run pruned early instead survive on other workers, raising the number generated (variable-dependent: +2% for y_4, +84% for y_9).
- **Subsumption time increased.** The redundancy removal that the serial run performs continuously during generation is deferred to one final pass, which is single-threaded and scales roughly with the square of the pile size (y_4: 3m28s → 29m42s).

## Tooling added
- Per-variable timing table with a speedup column.
- Single-variable isolation benchmark.
- Raw-resolvent dump for inspection.

## Pending
- Validate the incremental tautology stop: `python3 -m pytest tests/test_resolution.py -q` (expect `25 passed`).
- 9×9 compile (100 cores) in progress; feasibility unconfirmed.
