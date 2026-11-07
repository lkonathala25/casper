# CASPER

**CASPER: A Projection-Based Neurosymbolic Layer for Scalable & Guaranteed Constraint Satisfaction** (NeurIPS 2026).
Lohith Konathala, Luca Andolfi, Eleonora Giunchiglia, Imperial College London.

> Many real-world problems demand exact constraint satisfaction to be meaningfully solved, yet existing neurosymbolic methods face a stark trade-off: those that guarantee satisfaction fail to scale, while those that scale must resort to approximation. Meanwhile, traditional deep learning approaches offer no satisfaction guarantees at all. In this paper, we propose CASPER, a neurosymbolic layer that simultaneously (i) guarantees constraint satisfaction, (ii) scales at inference time, (iii) is fully automated, requiring no problem-specific engineering, and (iv) is fully differentiable, making it composable with any neural predictor both at training and at inference time. CASPER casts constraint satisfaction as a closed-form Euclidean projection onto the constraint-feasible region, computable in a single forward pass. Experiments across MNIST arithmetic (up to 1024 digits), Sudoku solving, and Warcraft pathfinding show that CASPER is the only method that produces guaranteed-valid outputs at scale: on 30×30 Warcraft grids, exact baselines time out and approximate ones produce valid paths less than 7% of the time, while CASPER produces them 100% of the time.

This repository contains CASPER and the scripts to train and evaluate it on the three tasks.

## Layout

| Path | Contents |
|---|---|
| `casper/` | The layer: signed clauses, offline compilation by signed resolution (`resolution.py`), the closed-form projection (`minedit.py`), the cvxpy variant (`cvxpy_minedit.py`) and the training objective (`casper_loss.py`) |
| `tasks/mnist/` | MNIST-Sum and MNIST-Add |
| `tasks/warcraft/` | Warcraft pathfinding (steps encoding, walk decoder, evaluation) |
| `tasks/sudoku/` | Visual Sudoku (SATNet) |
| `constraints/` | Compiled constraint artifacts that are small enough to ship |
| `tools/runlib.sh` | Shared helpers for the run scripts |

## Setup

```bash
conda create -n casper python=3.11 -y && conda activate casper
pip install -r requirements.txt
```

Datasets are not in the repository. Each section below says where to get them. Run outputs go to `runs/` (set `RUNS_ROOT` to change this). All run scripts take `SEEDS` (default `0 1 2 3 4`; Sudoku `0`–`9`) and skip runs that have already finished.

`MODE` and `L2_PENALTY` select the CASPER variant in both the MNIST and Warcraft scripts:

| Variant | Setting |
|---|---|
| CASPER | `MODE=train_infer L2_PENALTY=<λ>` |
| CASPER (w/o ℓ2) | `MODE=train_infer L2_PENALTY=0` |
| CASPER (Inference Only) | `MODE=infer_only` |
| CASPER (Train Only) | read from the CASPER checkpoints at evaluation time |

## MNIST-Sum and MNIST-Add

MNIST is downloaded automatically to `data/`. The constraint files are generated and compiled on first use.

```bash
# CASPER (λ = 1), MNIST-Sum n ∈ {2,16,64,256,1024} and MNIST-Add n ∈ {1,2,4,15,100,256}
MODE=train_infer L2_PENALTY=1 bash tasks/mnist/run_casper.sh

# CASPER (cvxpy): the same projection solved by SCS
MODE=train_infer L2_PENALTY=1 BACKEND=cvxpy bash tasks/mnist/run_casper.sh

# One setting only
MODE=train_infer L2_PENALTY=1 TASKS=sum SUM_NS=64 SEEDS=0 bash tasks/mnist/run_casper.sh
```

`tasks/mnist/launch_casper.sh` takes the same variables and starts one detached process per seed on `GPU`. Each run writes `results.json` with per-epoch metrics.

## Warcraft pathfinding

1. **Data.** Download the Warcraft shortest-path dataset of Vlastelica Pogančić et al. (2020) (linked from [martius-lab/blackbox-backprop](https://github.com/martius-lab/blackbox-backprop)). Put the raw `{train,val,test}_{maps,vertex_weights,shortest_paths}.npy` files in `data/warcraft/12x12/` and `data/warcraft/30x30/`. Then derive the 4-connected labels:

   ```bash
   python tasks/warcraft/data_prep/warcraft_preprocess.py --data_dir data/warcraft/12x12 --grid_size 12
   python tasks/warcraft/data_prep/warcraft_preprocess.py --data_dir data/warcraft/30x30 --grid_size 30
   ```

2. **Constraints.** The compiled 12×12 steps encoding ships in `constraints/warcraft/steps_artifacts/`. The 30×30 one (about 600 MB) has to be built once (about 16 minutes):

   ```bash
   python tasks/warcraft/build_steps_artifacts.py --size 30
   python tasks/warcraft/compile_steps_encoding.py constraints/warcraft/steps_artifacts/steps_30_trimmed.txt --no-verbose-pass
   ```

3. **Train** (ResNet-18, 300 epochs, early stopping after at least 100):

   ```bash
   MODE=train_infer L2_PENALTY=2 GRIDS="12 30" bash tasks/warcraft/run_casper.sh   # CASPER
   MODE=train_infer L2_PENALTY=0 GRIDS="12 30" bash tasks/warcraft/run_casper.sh   # w/o ℓ2
   MODE=infer_only               GRIDS="12 30" bash tasks/warcraft/run_casper.sh   # Inference Only
   MODE=train_infer L2_PENALTY=1 GRIDS="12 30" bash tasks/warcraft/run_casper.sh   # source of the Train Only row
   ```

4. **Evaluate.** This computes optimal, ≤1/5/10%, exact match, Hamming and consistency on the loop-erased path, plus the cumulative cost-optimality curve:

   ```bash
   python tasks/warcraft/eval_warcraft_all.py --grid-size 12
   python tasks/warcraft/eval_warcraft_all.py --grid-size 30
   ```

## Visual Sudoku

The pipeline follows Cornelio et al. (2023): a CNN, then a board solver, then CASPER, then a mask network, then a Prolog solver. It needs:

- the SATNet visual Sudoku data and MNIST, under `data/sudoku/data/` (`SUDOKU_DATA_DIR`);
- the pretrained NASR weights `{perception,solvernn,mask}/satnet/checkpoint_best.pth`, under `data/sudoku/pretrained/` (`SUDOKU_PRETRAINED_DIR`);
- SWI-Prolog for `pyswip`. It is found in the active conda env by default (`SUDOKU_SWIPL_ENV`).

Run `python tasks/sudoku/visual/paths.py` to check that every location is found. Then:

```bash
# Board solver from Cornelio et al. (2023)
bash tasks/sudoku/visual/scripts/run_mask_calibrated.sh

# LLM board solver (Qwen2.5-14B-Instruct + SATNet LoRA adapter under data/sudoku/satnet_qwen_sft/)
pip install -r tasks/sudoku/visual/requirements-llm.txt
FILL_CACHE=1 bash tasks/sudoku/visual/scripts/run_llm_mask_calibrated.sh
```

`FILL_CACHE=1` generates the LLM's completed boards once and caches them in `data/sudoku/llm_cache/`. Later runs read from the cache.
