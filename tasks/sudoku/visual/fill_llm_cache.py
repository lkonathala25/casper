"""Populate the LLM adapter's disk cache for every board of the given splits.

Runs the frozen perception CNN over each split, finds the boards whose
perception-argmax puzzle is not yet cached, and generates only those with the
fine-tuned Qwen adapter. The adapter flushes the cache to disk after every
solve_batch call, so an interrupted run keeps all finished chunks.

Boards whose LLM output fails to parse are never cached (by adapter design),
so a split can stay below 100% coverage; the count is reported at the end.

Usage (sudoku-fine env, GPU; the cache is read-only for every other script,
see paths.LLM_CACHE_DIR):
    python fill_llm_cache.py [--llm-cache-dir DIR]   # default paths.LLM_CACHE_DIR
"""
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

# Same SWI-Prolog/CUDA environment handling as the training scripts is not
# needed here (no Prolog), but the adapter import must happen before torch
# work for its libcudart/datasets workarounds.
from models.llm_solver_adapter import LLMSolverAdapter, DEFAULT_FINETUNED_MODEL_PATH

import torch

import paths

from datasets import SudokuDataset_RL
from models.transformer_sudoku import get_model


def main():
    p = ArgumentParser('fill_llm_cache')
    p.add_argument('--data', default='satnet')
    p.add_argument('--splits', default='-test,-valid,-train')
    p.add_argument('--llm-cache-dir', default=str(paths.LLM_CACHE_DIR))
    p.add_argument('--finetuned-model-path', default=None)
    p.add_argument('--generation-batch-size', default=4, type=int)
    p.add_argument('--chunk', default=32, type=int,
                   help='Boards per solve_batch call (cache is flushed after each).')
    args = p.parse_args()

    device = torch.device('cuda')
    model = get_model(block_len=81)
    model.load_pretrained_models(args.data)
    model.to(device).eval()

    adapter = LLMSolverAdapter(
        model_path=args.finetuned_model_path or DEFAULT_FINETUNED_MODEL_PATH,
        cache_dir=args.llm_cache_dir,
        generation_batch_size=args.generation_batch_size,
    )

    try:
        for split in [s.strip() for s in args.splits.split(',') if s.strip()]:
            ds = SudokuDataset_RL(args.data, split)
            dl = torch.utils.data.DataLoader(ds, batch_size=500, shuffle=False)
            with torch.no_grad():
                percs = torch.cat([model.perception(x.to(device)).cpu() for x, _ in dl])
            n = percs.shape[0]
            hit = set(adapter.get_cached_indices(percs))
            missing = [i for i in range(n) if i not in hit]
            print(f'[fill] {split}: {len(hit)}/{n} cached, generating {len(missing)}', flush=True)

            t0 = time.time()
            for start in range(0, len(missing), args.chunk):
                idx = missing[start:start + args.chunk]
                adapter.solve_batch(percs[idx].to(device))
                done = start + len(idx)
                rate = done / max(time.time() - t0, 1e-6)
                print(f'[fill] {split}: {done}/{len(missing)} generated '
                      f'({rate:.2f} boards/s, parse failures so far '
                      f'{adapter._parse_fail_count})', flush=True)

            final = len(adapter.get_cached_indices(percs))
            print(f'[fill] {split}: DONE {final}/{n} cached '
                  f'({n - final} uncached = parse failures)', flush=True)
    finally:
        adapter.close()


if __name__ == '__main__':
    main()
