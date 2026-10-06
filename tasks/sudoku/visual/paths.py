"""Every filesystem location used by the visual-Sudoku pipeline.

Each value is read from its environment variable, falling back to the default
below. The defaults point at the existing assets on the la2324 machine, so
nothing large is copied into the repo (see the root README for the data table).

    python paths.py            # print every location and whether it exists
    python paths.py --shell    # NAME=value lines for `eval` in the run scripts
"""
import os
import shlex
import sys
from pathlib import Path

VISUAL_DIR = Path(__file__).resolve().parent
CASPER_ROOT = VISUAL_DIR.parents[2]


def _path(var, default):
    return Path(os.environ.get(var, str(default)))


# Completed-board cache of the SFT LLM (read-only; only fill_llm_cache.py writes).
LLM_CACHE_DIR = _path('SUDOKU_LLM_CACHE_DIR', '/data/la2324/sudoku/llm_cache')
# SATNet visual Sudoku (satnet/) and MNIST (MNIST/raw, loaded by sudoku_solver/board.py).
DATA_DIR = _path('SUDOKU_DATA_DIR', '/home/la2324/llm-sudoku/sudoku/src/data')
# NASR pretrained weights: {perception,solvernn,mask,rl}/<data>/checkpoint_best.pth
PRETRAINED_DIR = _path('SUDOKU_PRETRAINED_DIR', '/home/la2324/llm-sudoku/sudoku/outputs')
# SATNet SFT LoRA adapter. Kept as the exact string: LLMSolverAdapter hashes it
# into every cache key, so any change (even a trailing slash) misses the cache.
LLM_SFT_PATH = os.environ.get(
    'SUDOKU_LLM_SFT_PATH',
    '/home/la2324/llm-sudoku/my_finetuning/outputs/satnet_qwen_sft/final',
)
# Checkpoints written by the run scripts.
CKPT_DIR = _path('SUDOKU_CKPT_DIR', VISUAL_DIR / 'checkpoint')
# llm-sudoku checkpoints used as read-only parity anchors (spec §2.3).
REFERENCE_CKPT_DIR = _path('SUDOKU_REFERENCE_CKPT_DIR',
                           '/home/la2324/llm-sudoku/sudoku/checkpoint')
# Training / evaluation logs written by the run scripts.
RESULTS_DIR = _path('SUDOKU_RESULTS_DIR', VISUAL_DIR / 'results')
# Conda env whose SWI-Prolog pyswip loads (LIBSWIPL_PATH / SWI_HOME_DIR).
SWIPL_ENV = _path('SUDOKU_SWIPL_ENV', '/home/la2324/miniconda3/envs/sudoku')
# Optional libcudart.so.12 shim for CUDA-12 wheels on CUDA-13 hosts.
CUDART12_DIR = _path('SUDOKU_CUDART12_DIR', '/home/la2324/ollama-bin/lib/ollama/cuda_v12')
# Compiled CASPER constraints for one ALLDIFFERENT group (committed).
GROUP_CONSTRAINTS = _path('SUDOKU_GROUP_CONSTRAINTS',
                          CASPER_ROOT / 'constraints' / 'sudoku' / 'sudoku_group9_compiled.txt')

NAMES = (
    'LLM_CACHE_DIR', 'DATA_DIR', 'PRETRAINED_DIR', 'LLM_SFT_PATH', 'CKPT_DIR',
    'REFERENCE_CKPT_DIR', 'RESULTS_DIR', 'SWIPL_ENV', 'CUDART12_DIR', 'GROUP_CONSTRAINTS',
)


if __name__ == '__main__':
    if '--shell' in sys.argv[1:]:
        for name in NAMES:
            print(f'{name}={shlex.quote(str(globals()[name]))}')
    else:
        for name in NAMES:
            value = globals()[name]
            print(f'{name:20s} {"ok     " if Path(value).exists() else "MISSING"} {value}')
