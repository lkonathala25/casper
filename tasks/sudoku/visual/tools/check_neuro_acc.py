"""NeuroAcc of the frozen pretrained solver under each edit layer (spec §4.6).

NeuroAcc is the board accuracy of argmax(edit_layer(solver_probs)), before the
mask and Prolog. It does not depend on mask training, so it checks the edit
layer and the solver before anything is retrained.

    python tools/check_neuro_acc.py --device cpu --layers legacy,casper,kpass24

Layers:
    legacy   llm-sudoku's Hungarian SudokuConstraintLayer (--edit-layer minedit)
    casper   CasperSudokuLayer = casper's GroupKPassCasper (--edit-layer casper)
    casper-search  the same layer with --casper-group-solver search
    kpass24  casper's original 24-peer KPassCasper

Expected on satnet -test, identical on CPU and GPU: legacy 330/500 (sat 0.9619),
casper 310/500 (sat 0.9530 on CPU; 317 certified), kpass24 286/500 (sat 0.9152,
293 certified).
"""
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

VISUAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISUAL_DIR))

import torch                                                        # noqa: E402
import torch.nn.functional as F                                     # noqa: E402

from casper_sudoku_layer import (                                   # noqa: E402
    DEFAULT_CONSTRAINTS_FILE, CasperSudokuLayer, load_group_constraints,
)
from datasets import SudokuDataset_RL                               # noqa: E402
from legacy.sudoku_constraint_layer import SudokuConstraintLayer    # noqa: E402
from local_casper import KPassCasper                                # noqa: E402  (casper)
from models.transformer_sudoku import get_model                     # noqa: E402

LAYERS = ('legacy', 'casper', 'casper-search', 'kpass24')


def main():
    p = ArgumentParser('check_neuro_acc')
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--layers', default='legacy,casper,kpass24',
                   help=f'Comma-separated subset of {",".join(LAYERS)}.')
    p.add_argument('--data', default='satnet')
    p.add_argument('--split', default='-test', choices=['-train', '-valid', '-test'])
    p.add_argument('--rounds', default=10, type=int,
                   help='Iterations (legacy) / rounds (casper, kpass24).')
    p.add_argument('--batch-size', default=100, type=int)
    p.add_argument('--constraints-file', default=str(DEFAULT_CONSTRAINTS_FILE))
    args = p.parse_args()

    layers = [x.strip() for x in args.layers.split(',') if x.strip()]
    unknown = sorted(set(layers) - set(LAYERS))
    if unknown:
        p.error(f'unknown layers {unknown}; choose from {LAYERS}')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('--device cuda requested but CUDA is not available')
    dev = torch.device(args.device)

    model = get_model(block_len=81)
    model.load_pretrained_models(args.data)
    model.to(dev).eval()
    loader = torch.utils.data.DataLoader(
        SudokuDataset_RL(args.data, args.split), batch_size=args.batch_size, shuffle=False,
    )

    # Every layer's AllDiff sat uses the same definition (legacy layer's method).
    sat_ref = SudokuConstraintLayer(9).to(dev)
    built = {}
    if 'legacy' in layers:
        built['legacy'] = sat_ref
    if 'casper' in layers:
        built['casper'] = CasperSudokuLayer(
            constraints_path=args.constraints_file, max_rounds=args.rounds,
        ).to(dev)
    if 'casper-search' in layers:
        built['casper-search'] = CasperSudokuLayer(
            constraints_path=args.constraints_file, max_rounds=args.rounds,
            group_solver='search',
        ).to(dev)
    if 'kpass24' in layers:
        built['kpass24'] = KPassCasper(
            load_group_constraints(args.constraints_file), 9, max_rounds=args.rounds,
        )

    correct = {k: 0 for k in ['raw'] + layers}
    sat = {k: 0.0 for k in layers}
    cert = {k: 0 for k in layers}
    secs = {k: 0.0 for k in layers}
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev)
            gt = y.to(dev).argmax(-1) - 1
            B = x.shape[0]
            probs = F.softmax(model.nn_solver(torch.exp(model.perception(x))), -1)
            correct['raw'] += int((probs.argmax(-1) == gt).all(1).sum())
            for name in layers:
                t0 = time.time()
                if name == 'legacy':
                    out, info = built[name](probs, max_iterations=args.rounds)
                elif name in ('casper', 'casper-search'):
                    out, info = built[name](probs, max_iterations=args.rounds)
                    cert[name] += info['n_certified']
                else:
                    flat, info = built[name](probs.reshape(B, -1))
                    out = flat.view(B, 81, 9)
                    cert[name] += sum(c is not None for c in info['certified_at'])
                secs[name] += time.time() - t0
                correct[name] += int((out.argmax(-1) == gt).all(1).sum())
                sat[name] += sat_ref.constraint_satisfaction_rate(out) * B
            n += B

    print(f'[neuro-acc] data={args.data} split={args.split} device={args.device} '
          f'rounds={args.rounds} N={n}')
    print(f'[neuro-acc] raw solver argmax: {correct["raw"]}/{n} '
          f'({100 * correct["raw"] / n:.1f}%)')
    for name in layers:
        certified = str(cert[name]) if name != 'legacy' else '-'
        print(f'[neuro-acc] {name:8s} boards={correct[name]}/{n} '
              f'({100 * correct[name] / n:.1f}%) alldiff_sat={sat[name] / n:.4f} '
              f'certified={certified} time={secs[name]:.1f}s')


if __name__ == '__main__':
    main()
