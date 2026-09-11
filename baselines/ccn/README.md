# CCN+

Ported from `/home/lk1325/mnist_experiment/warcraft_crl/`. `warcraft/warcraft_ccn_layer.py`
is upstream apart from imports, paths and an `alphabet` argument. The training loop in
`warcraft/train_ccn.py` is extracted from `warcraft_crl_ablation.py --condition
ccn_plus_in_training`, which was never run on Warcraft upstream.

Both modes use CASPER's per-cell categorical head (`constraint_mode='none'`, full
alphabet) and the pishield propositional shield, which must be compiled -- resolution
along a variable order, plus pishield's strong-coherency rule expansion -- before it
can be applied.

## `--constraint steps` (default)

The shield's variables are the steps atoms `X_i = v`, one per (tick, cell) pair in
the compiled domains D_i of CASPER's steps artifact. Requirements
(`warcraft/steps_requirements.py`): at least one cell per tick, and `X_i = u` implies
`X_{i+1}` in the artifact's feasible set. The pinned endpoints are removed by unit
propagation (pishield rejects unit clauses). Pairwise at-most-one is off by default:
it is exponential to compile (160 s at 6x6), and without it a valid walk is still
guaranteed, since every true atom has a true successor and X_L = sink.

- **Compile once:** `warcraft/compile_steps_shield.py --grid-size N` writes the
  requirements, the compiled strata (`*_strata.pkl`) and the atom map to
  `constraints/warcraft/ccn/`; `train_ccn.py` loads the strata and refuses to run
  without them.
- **Link:** CASPER's time-aware bridge in torch (`steps_ccn.StepsLink`): the score of
  cell v at tick i is `P(v on path) * exp(-|d(s,v) - i|)`, `P(v on path) = 1 - P(skip)`.
- **Loss:** per-cell CE + `--lambda-req` (default 1.0) x ReqLoss on the steps atoms,
  with the ground-truth walk as the goal.
- **Eval:** greedy walk over the feasible sets by shielded score, loop erasure, edges.
  Raw per-cell argmax metrics are reported alongside. The shield runs with
  `iterative=False`, which pishield itself asserts is equivalent; at 12x12 it matched
  the default rule-by-rule forward exactly (max |diff| 0) at 0.3 s vs 16.7 s per batch.
- **Batch size:** 64 by default in this mode. The goal-conditioned shield builds a
  batch x active-rules x atoms tensor, about 0.14 GB per board at 12x12 (9.1 GB at
  64), so upstream's 128 does not fit on a 24 GB GPU.
- Checked on the 5x5 shield: ground-truth walks satisfy all requirements, every
  shielded output does too (raw inputs: none), every decoded walk is a valid path,
  and gradients reach the network.

**Compile cost** (last tick eliminated first). pishield's own subsumption step
(`ClausesGroup.compacted`) compares every pair of clauses after each elimination, which
makes the compile roughly cubic: 0.35 s at 5x5, 14.7 s at 7x7, 71 s at 8x8, 343 s at
9x9, and at 12x12 an elimination rate of 0.22 variables/s -- about 11 hours, so that
run was stopped. `compile_steps_shield.py` therefore swaps in an indexed subsumption
(`fast_compacted`) that only compares clauses sharing a literal. It returns the same
clause sets: the strata were identical rule for rule at 8x8 (61 strata, 1,901 rules;
11.9 s vs 71.1 s) and 9x9 (79 strata, 3,105 rules; 34.7 s vs 343.4 s).
At 12x12 (8,842 free atoms, 8,981 requirements from CASPER's compiled artifact) the
compile takes 467 s with 1.1 GB peak and gives 141 strata (1.3 MB pickled).
`--native-compaction` restores pishield's own step. Any CCN+ compile time reported
from this script is with the indexed subsumption. Peak memory stays low (0.5-0.6 GB
at 8x8-9x9).

## `--constraint local`

The shield's variables are the 868 per-cell categories, read through `sigmoid`, with
exactly-one-per-cell and neighbour-agreement clauses. These follow the grid, so the
compile blows up: 0.07 s at 3x3, 9.5 s at 4x4, over 180 s at 5x5; at 12x12 it was
stopped after 40 minutes with 200 of 868 variables done. Kept for reference.

The `warcraft_flow_constraints_ccnplus.txt` in `data/warcraft/12x12/preprocessed_4conn/`
is an old monotone (628-variable) requirements file and is not used.

**Dependency:** `pishield` 1.0.0 from PyPI, which imports `matplotlib` at load time.

**Run:** `run_warcraft.sh` -- compiles the steps shield if missing, then 2 runs x 300
epochs, lr 5e-4, patience 30 (upstream `run_crl_warcraft_ccn_plus.sh` defaults).
