# CASPER

CASPER compiles a constraint set into a differentiable layer: `casper/` holds the
shared compiler and layer (`constraints.py`, `resolution.py`, `casper_layer.py`,
the closed-form `minedit.py` MinEdit-L2 projection). `tasks/` holds the
benchmarks built on it (`mnist/`, `warcraft/`, `sudoku/`), and `baselines/`
holds the comparison methods, each with its own README. `tests/` is the CPU
test suite (`pytest tests/`). The MNIST-era README and requirements are kept
in `archive/`.

## Sudoku pipeline (`tasks/sudoku/visual`): data and setup

Visual Sudoku (SATNet boards rendered with MNIST digits): perception CNN →
neuro-solver → CASPER group-local edit layer (`GroupKPassCasper` in
`tasks/sudoku/local_casper.py`) → mask head → SWI-Prolog. Only the **mask head**
is trained, for two solver backbones: the standard Transformer, and the LLM
variant (Qwen2.5-14B-Instruct, 4-bit, SATNet LoRA SFT) with the fusion head
frozen. The detailed pipeline doc is
[`tasks/sudoku/visual/README.md`](tasks/sudoku/visual/README.md).

### 1. Required data and where it must be located

Every location is resolved in
[`tasks/sudoku/visual/paths.py`](tasks/sudoku/visual/paths.py) and can be
overridden by its environment variable. **Nothing in this table is committed
to git** except the group constraints file. `python tasks/sudoku/visual/paths.py`
prints every location and whether it exists.

| Asset | Files the code reads | Size | `paths.py` / env var | Default location | Needed for |
|---|---|---|---|---|---|
| SATNet visual Sudoku | `satnet/satnet_imgs-{train,valid,test}.hdf5`, `satnet/satnet_sol-{train,valid,test}.npy`, `satnet/satnet-test.npy` (puzzle hints, eval diagnostics); `MNIST/raw/` (loaded at import by `sudoku_solver/board.py`) | 4.8 GB (+64 MB MNIST) | `DATA_DIR` / `SUDOKU_DATA_DIR` | `/home/la2324/llm-sudoku/sudoku/src/data` | both |
| Pretrained NASR weights | `{perception,solvernn,mask}/satnet/checkpoint_best.pth` (`rl/` not needed) | 15 MB | `PRETRAINED_DIR` / `SUDOKU_PRETRAINED_DIR` | `/home/la2324/llm-sudoku/sudoku/outputs` | both |
| Group constraints | `constraints/sudoku/sudoku_group9_compiled.txt` (in repo; equals `to_dsl(from_json(sudoku_group9_compiled.json))`) | 62 KB | `GROUP_CONSTRAINTS` / `SUDOKU_GROUP_CONSTRAINTS` | repo | both |
| LLM completed-board cache | `llm_cache_sft_fullboard.json` (9821 boards, sha256 `b67dffe6…07a3`) | 4.5 MB | `LLM_CACHE_DIR` / `SUDOKU_LLM_CACHE_DIR` | **`/data/la2324/sudoku/llm_cache`** | LLM |
| SATNet SFT LoRA adapter | `adapter_config.json`, `adapter_model.safetensors`, tokenizer files | 274 MB | `LLM_SFT_PATH` / `SUDOKU_LLM_SFT_PATH` (**the path string must not change**: it is part of the cache key) | `/home/la2324/llm-sudoku/my_finetuning/outputs/satnet_qwen_sft/final` | LLM |
| Base LLM | `unsloth/qwen2.5-14b-instruct-unsloth-bnb-4bit` (Hugging Face) | 11 GB | Hugging Face cache (`HF_HOME`) | `~/.cache/huggingface/hub/` | LLM |
| SWI-Prolog | `lib/swipl/lib/x86_64-linux/libswipl.so` (tested with 10.0.0) | — | `SWIPL_ENV` / `SUDOKU_SWIPL_ENV` | `/home/la2324/miniconda3/envs/sudoku` | both |
| CUDA 12 runtime shim (**optional**) | `libcudart.so.12`, only for CUDA-12 wheels on CUDA-13 hosts. The adapter loads it only if present; **it is absent on this machine**, and the LLM runs worked without it. | — | `CUDART12_DIR` / `SUDOKU_CUDART12_DIR` | `/home/la2324/ollama-bin/lib/ollama/cuda_v12` | LLM (optional) |
| Reference checkpoints (parity gates only) | the four llm-sudoku checkpoints of the results table below | — | `REFERENCE_CKPT_DIR` / `SUDOKU_REFERENCE_CKPT_DIR` | `/home/la2324/llm-sudoku/sudoku/checkpoint` | gates |

Run outputs go to `CKPT_DIR` (`SUDOKU_CKPT_DIR`, default
`tasks/sudoku/visual/checkpoint/`) and `RESULTS_DIR` (`SUDOKU_RESULTS_DIR`,
default `tasks/sudoku/visual/results/`); both are git-ignored.

**The LLM cache is read-only.** Training and evaluation read completed boards
from it; no LLM generation is needed. The only writer is `fill_llm_cache.py`,
which `scripts/run_llm_mask_calibrated.sh` runs only with `FILL_CACHE=1`.
Running `evaluate_checkpoints.py --llm` or `rl_eval_sudoku.py --solver-backbone llm`
**without `--skip-cache-miss`** would generate the uncached boards and rewrite
the cache file (and write an `.eval_cache_built` sentinel next to it), so
always pass `--skip-cache-miss` (the run scripts do). For the key format and
the coverage per split, see `/data/la2324/sudoku/README.md`. That README calls
the model Qwen2.5-7B; the adapter was in fact trained on the 14B 4-bit base
above (see its `adapter_config.json`).

### 2. Environments

| Env | Used for | Tested versions |
|---|---|---|
| `sudoku` | Transformer variant, gates, tests | Python 3.11, torch 2.11.0, numpy 2.3.5, scipy 1.17.1, h5py 3.16.0, pyswip 0.3.3 (native SWI-Prolog 10.0.0) |
| `sudoku-fine` | LLM variant | Python 3.11, torch 2.10.0, unsloth 2026.4.8, transformers 5.5.0, peft 0.19.1, bitsandbytes 0.49.2, pyswip 0.3.2, using the `sudoku` env's SWI-Prolog through `SUDOKU_SWIPL_ENV` (`LIBSWIPL_PATH` / `SWI_HOME_DIR` are set at import) |

- The LLM variant needs a CUDA GPU: the adapter's constructor loads the 4-bit
  model even when every board comes from the cache.
- Neither env has pytest. Run the pipeline tests with the shim, without
  installing anything:
  `python tools/pytest_shim.py test_casper_sudoku_layer.py` (from `tasks/sudoku/visual`).
- For a new environment, `requirements.txt` at the repo root lists casper's core
  dependencies plus the Sudoku and LLM ones, with floors from the tested envs.

### 3. How to run

From `tasks/sudoku/visual`, both runs detached:

```bash
mkdir -p results/mask_calibrated results/llm_mask_calibrated

# Transformer (sudoku env), 10 seeds
SEEDS="0 1 2 3 4 5 6 7 8 9" GPU_ID=0 \
  setsid nohup scripts/run_mask_calibrated.sh > results/mask_calibrated/driver.log 2>&1 < /dev/null &

# LLM, fusion head frozen (sudoku-fine env), 5 seeds
SEEDS="0 1 2 3 4" GPU_ID=0 \
  setsid nohup scripts/run_llm_mask_calibrated.sh > results/llm_mask_calibrated/driver.log 2>&1 < /dev/null &
```

`PYTHON_BIN` selects the interpreter (default: the `sudoku` / `sudoku-fine`
env next to `SUDOKU_SWIPL_ENV`). Outputs:

- Transformer checkpoints and sidecars:
  `$CKPT_DIR/casper/mask_calibrated/e10_lr1e-4_wd3e-1_bs200_cos_wu0_nw20_clip1_it3-10_pat5_d1_bs0_-1_-2_casper/checkpoint_best_R_<seed>.{pth,json}`;
  test summary `evaluation_summary.log`, `evaluation_per_seed.csv` and
  `evaluation_raw.log` in the same folder.
- LLM: `$CKPT_DIR/casper/llm_mask_calibrated/seed<seed>_frozenhead/`; summary
  `$CKPT_DIR/casper/llm_mask_calibrated/evaluation_summary__frozenhead.log`.
- Training and evaluation logs: `$RESULTS_DIR/{mask_calibrated,llm_mask_calibrated}/`.

GPU parity gates (run before the full runs): `scripts/verify_gpu_gates.sh`.

### 4. Results so far (satnet `-test`, N=500)

NeuroAcc = Neuro-Solver board acc (%), RecRate = perception errors corrected
by the SolverNN (%), PipeAcc = NASR board acc (%).

| Run (llm-sudoku checkpoint) | Bias | NeuroAcc | RecRate | PipeAcc | AllDiff sat | Certified |
|---|---|---|---|---|---|---|
| Transformer, CASPER, seed 0 | −1 | 62.0 | 93.26599 | 82.6 | 0.9534 | 317 |
| LLM frozen head, CASPER, seed 0 | −1 | 62.0 | 93.92936 | 80.0 | 0.9530 | 317 |
| LLM frozen head, CASPER, seed 1 | −1 | 62.0 | 93.44484 | 81.0 | 0.9530 | 317 |
| Legacy MinEdit (Hungarian), seed 0, `--edit-layer minedit` | 0 | 66.0 | 93.30 | 84.8 | 0.9619 | — |
| *(target, legacy, 10 seeds)* | | 66.0 ± 0.0 | 94.3 ± 0.8 | 84.4 ± 0.9 | | |

NeuroAcc depends only on the solver and the edit layer: on the frozen
pretrained solver, the legacy layer gets 330/500, group-local CASPER 310/500,
and casper's 24-peer `KPassCasper` 286/500 (`tools/check_neuro_acc.py`).
