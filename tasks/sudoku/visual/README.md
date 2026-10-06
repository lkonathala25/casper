# Visual Sudoku: CASPER mask-only training

Port of the llm-sudoku MinEdit pipeline with casper's group-local edit layer.
Spec: `/home/la2324/llm-sudoku/migration/CASPER_MASK_TRAINING_PORT.md`.
Background: `/home/la2324/llm-sudoku/MINEDIT_PIPELINE_INDEX.md` (it predates the
CASPER layer; where the two disagree, the spec wins). Data locations, envs and
the requirements are in the [root README](../../../README.md).

```
images ─ perception CNN (frozen) ─┬─ Transformer: nn_solver (frozen) ─ softmax ─────────────┐
                                  └─ LLM: nn_solver + cached Qwen board + perception         │
                                          ─ fusion head (FROZEN, = solver probs at init) ───┤
                                                                                             ▼
   CASPER edit layer: CasperSudokuLayer → casper's GroupKPassCasper (no parameters)
                                                                                             ▼
   mask_nn (TRAINED) → cleaned = argmax([1-p, P·p]) (KEEP=1) → SWI-Prolog → final board
```

Only `mask_nn` is trained in both variants. The mask is trained and evaluated on
the **projected** distribution it gates; the target is `1[argmax(projected) == gt]`.

## Files

| File | Role |
|---|---|
| `paths.py` | Every filesystem location (env-var overrides); `python paths.py` lists them |
| `casper_sudoku_layer.py` | `CasperSudokuLayer` (SudokuConstraintLayer API around `GroupKPassCasper`), constraint loader and validation, `add_edit_layer_args`, `build_edit_layer` |
| `minedit_mask_calibrate.py` | Transformer, mask only |
| `minedit_llm_mask_calibrate.py` | LLM, mask only: `--freeze-fusion-head` is **on by default**; `--train-fusion-head` opts out with a warning. `--skip-cache-miss` is on by default (`--no-skip-cache-miss` opts out) |
| `rl_eval_sudoku.py` | One checkpoint on `-test`, prints every metric (including `CASPER certified: n/N`) |
| `evaluate_checkpoints.py` | Multi-seed driver; regex-parses the eval stdout; default `--edit-layer casper` (legacy checkpoints need `--edit-layer minedit`) |
| `fill_llm_cache.py` | Generates only the boards missing from the cache. **Writes the cache**; run it only on purpose (`FILL_CACHE=1`) |
| `legacy/` | Hungarian `SudokuConstraintLayer` + `VectorisedMinEditL2`, used only with `--edit-layer minedit` (A/B and parity) |
| `models/`, `datasets.py`, `sudoku_data.py`, `nasr.py`, `sudoku_solver/`, `utils/`, `group_order_utils.py` | Unchanged from llm-sudoku apart from paths |
| `tools/check_neuro_acc.py` | NeuroAcc of the frozen solver per edit layer (`--device cpu|cuda --layers legacy,casper,kpass24`) |
| `tools/check_cache_coverage.py` | LLM cache coverage per split, from perception argmax keys (no LLM) |
| `tools/pytest_shim.py` | Runs the tests in envs without pytest |
| `scripts/run_mask_calibrated.sh` | Transformer: seeds → train → `-test` eval (detached-safe) |
| `scripts/run_llm_mask_calibrated.sh` | LLM frozen head: cache coverage check → seeds → train → eval |
| `scripts/verify_gpu_gates.sh` | GPU parity gates P1–P5 |

The script names, CLI flags, stdout lines and sidecar fields match llm-sudoku,
so the Index and the reference logs map one to one. The exceptions are the
defaults above, the location defaults (now from `paths.py`), and the removed
`--prompt-path` / `--prompt-template` / `--llm-batch-size` flags, which the
adapter ignored.

## Run

All commands run from this directory. Detached full runs:

```bash
mkdir -p results/mask_calibrated results/llm_mask_calibrated
SEEDS="0 1 2 3 4 5 6 7 8 9" GPU_ID=0 \
  setsid nohup scripts/run_mask_calibrated.sh > results/mask_calibrated/driver.log 2>&1 < /dev/null &
SEEDS="0 1 2 3 4" GPU_ID=0 \
  setsid nohup scripts/run_llm_mask_calibrated.sh > results/llm_mask_calibrated/driver.log 2>&1 < /dev/null &
```

Env vars: `SEEDS`, `GPU_ID`, `PYTHON_BIN`, `EXTRA_ARGS` (appended to the training
command, e.g. `--epochs 1`), and `FILL_CACHE=1` (LLM only). Logs go to
`$RESULTS_DIR/<variant>/`. Checkpoints go to `$CKPT_DIR/casper/...`, with the test
summary next to them (see the root README).

Single checkpoint (Transformer; the bias comes from the sidecar's `best_bias`):

```bash
python rl_eval_sudoku.py --ckpt-path <ckpt.pth> --nasr rl --solver prolog --data satnet \
  --use-minedit --edit-layer casper --mask-logit-bias <best_bias>
```

LLM: add `--solver-backbone llm --use-perception-features --temp-hidden-dim 128 --skip-cache-miss`.
Always pass `--skip-cache-miss` with the LLM: without it the uncached boards are
generated and the cache file is rewritten.

## Checks

| Check | Command | Expected |
|---|---|---|
| Edit-layer unit tests (casper) | `python -m pytest -q tasks/sudoku/test_group_kpass.py` (repo root) | pass, incl. bit-identical parity with llm-sudoku |
| Adapter tests | `python tools/pytest_shim.py test_casper_sudoku_layer.py` | 9 passed, 1 skipped (CUDA) on CPU |
| NeuroAcc | `python tools/check_neuro_acc.py --device cpu` | legacy 330/500 (sat 0.9619), casper 310/500 (sat 0.9530, 317 certified), kpass24 286/500 (sat 0.9152, 293 certified) |
| Cache coverage | `python tools/check_cache_coverage.py --device cpu` | train 8847/9000, valid 489/500, test 482/500 (GPU perception: 8847/491/483) |
| GPU parity | `scripts/verify_gpu_gates.sh` | P1–P5 PASS |

## Reference results (llm-sudoku checkpoints, satnet `-test`, N=500)

| Run | Bias | NeuroAcc | RecRate | PipeAcc | AllDiff sat | Certified |
|---|---|---|---|---|---|---|
| Transformer, CASPER, seed 0 | −1 | 62.0 | 93.26599 | 82.6 | 0.9534 | 317 |
| LLM frozen head, CASPER, seed 0 | −1 | 62.0 | 93.92936 | 80.0 | 0.9530 | 317 |
| LLM frozen head, CASPER, seed 1 | −1 | 62.0 | 93.44484 | 81.0 | 0.9530 | 317 |
| Legacy MinEdit (Hungarian), seed 0, `--edit-layer minedit` | 0 | 66.0 | 93.30 | 84.8 | 0.9619 | — |
| *(target, legacy, 10 seeds)* | | 66.0 ± 0.0 | 94.3 ± 0.8 | 84.4 ± 0.9 | | |

Frozen at its identity init, the fusion head outputs exactly the solver's
probabilities, so in this configuration the LLM does not change the board. The
LLM variant differs from the Transformer only in its recipe and in training
and validating on the cached boards. Training the head jointly degraded it
(fused→CASPER valid accuracy fell from 58.0% to 27.3% over 9 epochs; PipeAcc 79.6).
