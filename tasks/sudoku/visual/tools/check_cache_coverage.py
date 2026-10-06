"""LLM cache coverage check (spec §4.5 pre-check, gate G5). Needs no LLM.

Runs the frozen perception CNN over each split, builds the adapter's cache key
from the perception argmax (LLMSolverAdapter._make_cache_key on a stub whose
model_path is paths.LLM_SFT_PATH), and counts the boards present in the cache.
Read-only: the cache file is only opened for reading.

    python tools/check_cache_coverage.py --device cuda --min-train 8847 --min-test 483

Measured (2026-10-06, cache sha256 b67dffe6...07a3):
    GPU perception: train 8847/9000, valid 491/500, test 483/500
    CPU perception: train 8847/9000, valid 489/500, test 482/500
Exit code 1 if a --min-* threshold is not met.
"""
import hashlib
import json
import sys
import types
from argparse import ArgumentParser
from pathlib import Path

VISUAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISUAL_DIR))

import torch                                                     # noqa: E402

import paths                                                     # noqa: E402
from datasets import SudokuDataset_RL                            # noqa: E402
from models.llm_solver_adapter import LLMSolverAdapter           # noqa: E402
from models.transformer_sudoku import get_model                  # noqa: E402

CACHE_FILE = 'llm_cache_sft_fullboard.json'


def main():
    p = ArgumentParser('check_cache_coverage')
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--data', default='satnet')
    p.add_argument('--llm-cache-dir', default=str(paths.LLM_CACHE_DIR))
    p.add_argument('--splits', default='-train,-valid,-test')
    for split in ('train', 'valid', 'test'):
        p.add_argument(f'--min-{split}', default=0, type=int,
                       help=f'Fail if fewer {split} boards are cached.')
    args = p.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('--device cuda requested but CUDA is not available')
    dev = torch.device(args.device)

    cache_path = Path(args.llm_cache_dir) / CACHE_FILE
    raw = cache_path.read_bytes()
    cache = json.loads(raw)
    print(f'[cache] {cache_path}: {len(cache)} entries, '
          f'sha256 {hashlib.sha256(raw).hexdigest()}')
    print(f'[cache] key model_path = {paths.LLM_SFT_PATH}')

    stub = types.SimpleNamespace(model_path=paths.LLM_SFT_PATH)
    model = get_model(block_len=81)
    model.load_pretrained_models(args.data)
    model.to(dev).eval()

    failed = False
    for split in [s.strip() for s in args.splits.split(',') if s.strip()]:
        loader = torch.utils.data.DataLoader(
            SudokuDataset_RL(args.data, split), batch_size=500, shuffle=False,
        )
        hits = n = 0
        with torch.no_grad():
            for x, _ in loader:
                digits = model.perception(x.to(dev)).exp().argmax(dim=2).cpu().tolist()
                hits += sum(LLMSolverAdapter._make_cache_key(stub, b) in cache for b in digits)
                n += len(digits)
        need = getattr(args, f'min_{split.lstrip("-")}')
        ok = hits >= need
        failed |= not ok
        print(f'[cache] {split}: {hits}/{n} cached (device={args.device}'
              + (f', min {need}: {"PASS" if ok else "FAIL"}' if need else '') + ')')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
