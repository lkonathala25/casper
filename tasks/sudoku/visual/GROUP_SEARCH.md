# CASPER group solver: from greedy repair to search

This change adds a second way for the CASPER edit layer to repair a Sudoku
group: `--casper-group-solver search`. It closes the gap between the CASPER
pipeline and the legacy Hungarian MinEdit layer. The default stays `greedy`,
so every earlier result reproduces unchanged.

| Transformer, satnet `-test`, N=500 | NeuroAcc | PipeAcc |
|---|---|---|
| CASPER, greedy (previous) | 62.0 | 81.96 ± 0.56 (5 seeds) |
| **CASPER, search (new)** | **66.0** | **84.72 ± 0.72** (5 seeds) |
| Legacy MinEdit (Hungarian), README target | 66.0 | 84.4 ± 0.9 (10 seeds) |

---

## 1. Where the edit layer sits

```
images → perception CNN → solver → per-cell probabilities (81 cells × 9 digits)
                                              │
                                   CASPER edit layer  ◀── this change
                                              │
                         mask head (trained) → SWI-Prolog → final board
```

The solver gives each cell a probability for each digit. Its argmax board
usually breaks some Sudoku rules: two cells in a row, column or box hold the
same digit. The edit layer changes as little probability mass as possible so
that the argmax board satisfies the rules. NeuroAcc is the accuracy of that
repaired board. The mask head then decides which repaired cells to trust
before Prolog completes the board.

Both versions of the layer run the same outer loop. They sweep the 27 groups
(9 rows, 9 columns, 9 boxes) for up to 10 rounds, and a board counts as
*certified* after a round with no edits. The two versions differ only in how
they repair **one group** that contains a duplicate.

## 2. The previous implementation (`greedy`)

`GroupKPassCasper` repaired a group one cell at a time:

1. Visit the group's 9 cells, least confident first, where confidence is the
   probability of the cell's current digit.
2. If another cell of the group holds the current cell's digit, move the
   current cell to the nearest (MinEdit L2) digit that no other cell of the
   group holds.
3. The move is final for this pass. The cell isn't reconsidered when later
   cells are visited.

From the compiled constraint file, this rule only uses the pairwise entries:
"y_i ≠ v when y_j = v" (`_not_mask` in `local_casper.py`). The higher-order
clauses of `sudoku_group9_compiled.txt` were never consulted.

## 3. Why having all the compiled constraints is not enough

The compiled `.txt` file is complete. For any partial assignment of a group, it
tells you which values the next cell can take so that the group can still be
finished correctly. That guarantees **satisfiability**: if you assign cells one
by one and always pick from the allowed values, you never get stuck and always
end with a valid group.

It doesn't guarantee **optimality**. Many valid groups exist, and
the constraints don't say which one is most likely. A one-cell-at-a-time
procedure commits each cell before seeing the rest. So a cell can take a digit
that a later cell needed far more, and once committed, that choice is never
undone.

We checked this on the frozen pretrained solver (satnet `-test`, 500 boards).
All the variants below read the **same** compiled file. The only difference is
how they use it.

| How each group is repaired | NeuroAcc |
|---|---|
| Full CASPER pass over all compiled clauses, least-confident cell first | 151 / 500 |
| Full CASPER pass, canonical cell order | 244 / 500 |
| Current `GroupKPassCasper` (pairwise lookups, least-confident moves) | 310 / 500 |
| Full CASPER pass, most-confident cell first | 311 / 500 |
| **Search over the compiled feasible values (new)** | **330 / 500** |
| Legacy Hungarian (best permutation per group) | 330 / 500 |

The constraint knowledge is the same in every row, yet the greedy variants
range from 151 to 311 depending only on the order of the choices. What closes
the gap is choosing the most likely complete group instead of committing one
cell at a time.

## 4. The new implementation (`search`)

For a group that isn't yet a permutation, `GroupKPassCasper` with
`group_solver="search"` does the following:

1. **Cost table.** Build cost(cell, digit) = −log p(digit | cell) from the
   current probabilities, 9 × 9 per group. The probability of a whole
   assignment is the product of the per-cell probabilities, so the lowest
   total cost is the most likely assignment.
2. **Search over the compiled walk.** Assign the group's variables in the
   compiled file's own order (`y_1 … y_9`). At each step, only try the values
   the compiled file allows after the values placed so far
   (`constraints.feasible_set(step, prefix)`), cheapest first.
3. **Branch and bound.** Keep the best complete assignment found so far. Drop
   a partial assignment when its cost plus the cheapest possible cost of the
   remaining cells can't beat it. The bound never overestimates, so the
   result is the exact optimum.
4. **Projection.** Project each cell whose digit changed onto its new digit
   with the same positive-head MinEditL2 as before. Cells that keep their
   digit aren't touched.

In short, the old rule asked who is least sure of the contested digit; the new
one asks which complete assignment of the group is most likely.

Some practical details:

- **Caching.** Feasible sets are cached per prefix, so each distinct query to
  the compiled file is made once.
- **Speed.** About 2.8 s per 500 boards, against 2.9 s for greedy.
- **Fail-safe.** If the compiled file admits no complete assignment, the layer
  raises an error. Returning "no edits" instead would wrongly certify an
  invalid board.

### Relation to the legacy Hungarian layer

For AllDifferent, the result per group is the same assignment the Hungarian
algorithm picks: identical NeuroAcc (330) and AllDiff sat (0.9619). The
difference is where the constraints come from:

- **Hungarian** is specific to AllDifferent.
- **The search** reads the allowed values from the compiled CASPER file. It
  makes no assumption about the constraint beyond that file, so it works for
  any compiled constraint set.

## 5. Examples

### Example A: toy row

Row cells A and B both predict digit 3, and digit 5 is the only digit missing
from the row. One of A or B has to become 5.

| cell | p(3) | p(5) |
|---|---|---|
| A | 0.60 | 0.35 |
| B | 0.55 | 0.01 |

- **Greedy.** B is the less confident holder of 3 (0.55 < 0.60), so B is
  visited first and moves to the only free digit, 5. B gives 5 only 1%.
- **Search.** It compares the two complete options:
  - A→5, B=3: 0.35 × 0.55 = 0.19
  - A=3, B→5: 0.60 × 0.01 = 0.006

  It moves A, about 30× more likely.

This is the unit test `test_search_moves_the_cell_with_the_plausible_alternative`.

### Example B: a real board (satnet `-test` board 47, row 5)

These are the frozen solver's outputs for row 5 (digits shown 1–9). Cells 5
and 7 both predict 4, and the missing digit is 3.

| cell | solver digit (p) | p(3) | true digit | greedy | search |
|---|---|---|---|---|---|
| 1 | 1 (1.000) | | 1 | 1 | 1 |
| 2 | 8 (1.000) | | 8 | 8 | 8 |
| 3 | 5 (1.000) | | 5 | 5 | 5 |
| 4 | 2 (1.000) | | 2 | 2 | 2 |
| **5** | **4 (0.933)** | 0.0004 | 4 | **3** ✗ | **4** ✓ |
| 6 | 6 (1.000) | | 6 | 6 | 6 |
| **7** | **4 (0.968)** | 0.0308 | 3 | **4** ✗ | **3** ✓ |
| 8 | 7 (1.000) | | 7 | 7 | 7 |
| 9 | 9 (1.000) | | 9 | 9 | 9 |

- **Greedy.** Cell 5 is slightly less confident in its 4 (0.933 vs 0.968), so
  greedy moves it to 3, a digit it gives 0.04%. The row becomes valid but
  wrong in two cells, and the board is lost.
- **Search.** It compares:
  - cell 5 → 3: 0.0004 × 0.968 ≈ 0.0004
  - cell 7 → 3: 0.933 × 0.0308 ≈ 0.029

  The second option is about 70× more likely, so cell 7 moves. That's the
  true solution, and the board ends up fully correct.

Both cells are very sure of 4, so confidence in the contested digit can't
separate them. What separates them is that cell 7 has a plausible second
choice and cell 5 doesn't. Greedy never looks at second choices; the search
does.

Boards 108 (row 7) and 119 (box 8) behave the same way. Two cells share a
digit, greedy moves the less confident one and the board ends up wrong, while
search moves the other one and recovers the true solution.

### Example C: why the compiled walk is enough to search

When the search places digits in a row whose first three cells are already 1,
8 and 5, it asks the compiled file what `y_4` may be after (1, 8, 5). The
answer is {2, 3, 4, 6, 7, 9}. A branch that would repeat a digit is never
generated, so the search never builds an invalid group and never needs a
separate validity check. Because the file is complete, every branch it does
generate can be finished. The bound then prunes the unlikely branches. A
group therefore never needs all 9! = 362,880 permutations enumerated: the
whole 500-board test set takes about 2.8 s.

## 6. What did *not* change

- **Model and training.** Perception, solver, LLM adapter, fusion head, mask
  head, Prolog, and the training and evaluation recipes are all unchanged.
- **Constraints.** The compiled constraint file is unchanged.
- **Outer loop.** 27 groups, up to 10 rounds, certified after a round with
  zero edits.
- **Projection and gradients.** The per-cell MinEditL2 projection is
  unchanged, so gradients flow exactly as before. The search only decides
  which digit each cell is projected onto.
- **Checkpoints.** The layer has no parameters, so checkpoints are
  interchangeable. A mask head trained against one solver should still be
  evaluated with the same solver, because it learned that solver's output
  distribution.
- **Default.** `greedy` is the default everywhere. The bit-identical parity
  test against llm-sudoku covers that path, and the CPU NeuroAcc of `casper`,
  `legacy` and `kpass24` is unchanged (310 / 330 / 286).

## 7. Results (satnet `-test`, N=500, seeds 0–4)

| Run | NeuroAcc | RecRate | PipeAcc | AllDiff sat | Certified |
|---|---|---|---|---|---|
| Transformer, CASPER greedy | 62.0 | 94.05 ± 1.13 | 81.96 ± 0.56 | 0.9533 | 317 |
| **Transformer, CASPER search** | **66.0** | 94.21 ± 1.09 | **84.72 ± 0.72** | 0.9619 | **338** |
| LLM frozen head, CASPER greedy | 62.0 | 94.66 ± 0.63 | 80.12 ± 1.15 | 0.9530 | 317 |
| **LLM frozen head, CASPER search** | **66.0** | 94.68 ± 1.27 | **83.20 ± 1.54** | 0.9619 | **338** |
| Legacy MinEdit, README target (10 seeds) | 66.0 | 94.3 ± 0.8 | 84.4 ± 0.9 | 0.9619 | — |

PipeAcc per seed:
- **Transformer, search:** 84.6, 85.8, 84.0, 85.0, 84.2
- **LLM, search:** 84.0, 80.6, 83.2, 84.6, 83.6

The LLM evaluations were cache-only (`cache_hits=483 cache_misses=17`), and the
cache sha256 is unchanged (`b67dffe6…07a3`).

Notes:

- **Same NeuroAcc for both variants.** With the fusion head frozen, the LLM
  variant feeds the same probabilities into the edit layer as the
  Transformer. The difference in PipeAcc comes from the mask head, which is
  trained and validated on the cached LLM boards.
- **Fewer correct boards broken.** With the Transformer, the share of correct
  boards that NASR corrupts fell from 0.26% to 0.06%.
- **Seed counts differ.** The target row is a 10-seed mean and the new rows
  use 5 seeds.

## 8. How to use it

```bash
cd ~/casper/tasks/sudoku/visual && source /Data/la2324/sudoku/env.sh

# NeuroAcc of the frozen solver, all layers
python tools/check_neuro_acc.py --device cpu --layers legacy,casper,casper-search,kpass24

# Training + evaluation (both run scripts accept CASPER_GROUP_SOLVER)
CASPER_GROUP_SOLVER=search SEEDS="0 1 2 3 4" GPU_ID=0 \
  setsid nohup scripts/run_mask_calibrated.sh     > results/mask_calibrated_search.log 2>&1 < /dev/null &
CASPER_GROUP_SOLVER=search SEEDS="0 1 2 3 4" GPU_ID=1 \
  setsid nohup scripts/run_llm_mask_calibrated.sh > results/llm_mask_calibrated_search.log 2>&1 < /dev/null &

# Single checkpoint
python rl_eval_sudoku.py --ckpt-path <ckpt.pth> --nasr rl --solver prolog --data satnet \
  --use-minedit --edit-layer casper --casper-group-solver search --mask-logit-bias <best_bias>
```

Search runs write to their own folders, so greedy results are never
overwritten:

- **Checkpoints:**
  - Transformer: `checkpoint/casper/mask_calibrated/e10_…_casper_search/`
  - LLM: `checkpoint/casper/llm_mask_calibrated_search/`
- **Logs:** `results/mask_calibrated_search/` and
  `results/llm_mask_calibrated_search/`

The checkpoint sidecar JSON records `casper_group_solver`.

## 9. Files changed

| File | Change |
|---|---|
| `tasks/sudoku/local_casper.py` | `GroupKPassCasper(group_solver=...)`: `_compiled_feasible` (cached compiled-walk queries), `_search_group` (branch and bound), `_repair_group_search` (projection of changed cells) |
| `visual/casper_sudoku_layer.py` | `CasperSudokuLayer(group_solver=...)`; `--casper-group-solver` in `add_edit_layer_args`; passed through `build_edit_layer` and printed in its log line |
| `visual/evaluate_checkpoints.py` | Accepts `--casper-group-solver` and forwards it to `rl_eval_sudoku.py` |
| `visual/minedit_mask_calibrate.py`, `visual/minedit_llm_mask_calibrate.py` | Record `casper_group_solver` in the checkpoint sidecar |
| `visual/scripts/run_mask_calibrated.sh`, `visual/scripts/run_llm_mask_calibrated.sh` | `CASPER_GROUP_SOLVER` env var (default `greedy`), passed to training and evaluation; `_search` suffix on the output folders |
| `visual/tools/check_neuro_acc.py` | New layer option `casper-search` |
| `tasks/sudoku/test_group_kpass.py` | 6 new tests (see below) |

One visible change for greedy runs: the "Edit layer:" line of
`evaluation_summary*.log` now also lists `--casper-group-solver greedy`.

## 10. Tests

These run in `test_group_kpass.py` via `tools/pytest_shim.py`, all passing:

- **Invalid option:** an unknown solver name is rejected.
- **Valid board:** a valid board needs zero edits and is certified in round 1.
- **Single duplicate:** one duplicate is repaired back to the true board.
- **Gradients:** gradients flow through the search path.
- **Example A:** the cell with the plausible alternative is the one that
  moves.
- **Optimality:** on 20 random 9 × 9 cost tables, the search finds the same
  minimum cost as `scipy.optimize.linear_sum_assignment` (Hungarian).

`test_casper_sudoku_layer.py` is unchanged: 9 passed, 1 skipped (CUDA).

## 11. Limitations

- **Optimal per group, not per board.** Each group gets its most likely
  assignment, but a later group can still change a cell an earlier group
  set. Cross-group conflicts are resolved by the outer rounds, as with
  Hungarian. 162 of 500 boards are still not certified after 10 rounds.
- **Cost.** The search optimizes −log p, like Hungarian, while the
  projection minimizes L2. The two usually agree on the chosen digit but
  aren't guaranteed to.
- **Search size.** Branch and bound is exponential in the worst case. With
  peaked network outputs it explores few nodes, but very flat distributions
  over many conflicting cells would be slower.
- **Not re-checked on GPU.** The GPU parity gates (P1–P5) weren't re-run on
  this branch. They test the default `greedy` path, which is unchanged.
