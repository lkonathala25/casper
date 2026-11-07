"""CASPER edit layer for the 9x9 pipeline — drop-in for SudokuConstraintLayer.

The admissible assignments of one ALLDIFFERENT group are read from a compiled
CASPER constraint file (default: paths.GROUP_CONSTRAINTS, i.e.
constraints/sudoku/sudoku_group9_compiled.txt, the flattened
constraints/sudoku/sudoku_group9_compiled.json). The projection is casper's
single-cell MinEditL2 positive head onto the nearest feasible digit, as in
KPassCasper.

The edit layer itself is casper's GroupKPassCasper (tasks/sudoku/local_casper.py):
a cell's feasible set is computed from the 8 other cells of the group currently
being processed, not from all 24 row/col/box peers at once, and within a group
cells are visited least-confident first (cell_order "confidence").

Call contract matches SudokuConstraintLayer:
    adjusted, info = layer(cell_probs (B,81,9), max_iterations=None)
    layer.constraint_satisfaction_rate(cell_probs) -> float
"""
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn

import paths
from group_order_utils import validate_group_order, parse_group_order, format_group_order

CASPER_ROOT = Path(__file__).resolve().parents[3]
# Appended (not prepended) so visual/ modules (datasets, models, utils, ...) keep
# priority over casper's task modules.
for _sub in ('casper', 'tasks/sudoku'):
    _p = str(CASPER_ROOT / _sub)
    if _p not in sys.path:
        sys.path.append(_p)

from constraints import Constraints                                  # noqa: E402  (casper)
from local_casper import GroupKPassCasper, alldiff_group_constraints  # noqa: E402  (casper)
from sudoku_groups import build_groups                              # noqa: E402  (casper)

DEFAULT_CONSTRAINTS_FILE = paths.GROUP_CONSTRAINTS


def load_group_constraints(path, fmt='auto'):
    """Load one compiled group constraint set.

    fmt: 'auto' | 'compiled-txt' | 'json' | 'dsl'. 'auto' accepts .json and
    *_compiled.txt only: a raw DSL .txt would be compiled from scratch, which
    takes tens of minutes for the 9-clique, so it must be requested explicitly.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'constraints file not found: {path}')
    if fmt == 'auto':
        if path.suffix == '.json':
            fmt = 'json'
        elif path.name.endswith('_compiled.txt'):
            fmt = 'compiled-txt'
        else:
            raise ValueError(
                f'cannot infer the format of {path}; pass --constraints-format '
                f'compiled-txt (already compiled) or dsl (compile at load, slow)'
            )
    if fmt == 'json':
        return Constraints.from_file(path)
    if fmt == 'compiled-txt':
        return Constraints.from_compiled_dsl(path)
    if fmt == 'dsl':
        warnings.warn(f'compiling raw DSL {path} at load time; this can take a long time')
        return Constraints.from_file(path)
    raise ValueError(f'unknown constraints format {fmt!r}')


def validate_group_constraints(cs, grid_size=9):
    """The layer reads the file through pairwise queries feasible_set(1, {0: v})
    and applies them to every pair of cells in a group, so only clique-symmetric
    ALLDIFFERENT groups are meaningful."""
    domains = list(cs.var_domains)
    if domains != [grid_size] * grid_size:
        raise ValueError(
            f'constraints have domains {domains}; expected {grid_size} variables '
            f'of domain {grid_size} (one Sudoku group)'
        )
    oracle = alldiff_group_constraints(grid_size)
    for v in range(grid_size):
        got = cs.feasible_set(1, {0: v})
        want = oracle.feasible_set(1, {0: v})
        if got != want:
            raise ValueError(
                f'constraints are not a {grid_size}-clique ALLDIFFERENT group: '
                f'feasible_set(1, {{0: {v}}}) = {sorted(got)}, expected {sorted(want)}'
            )


class CasperSudokuLayer(nn.Module):
    """(B,81,9) wrapper around GroupKPassCasper with SudokuConstraintLayer's API."""

    def __init__(self, constraints_path=DEFAULT_CONSTRAINTS_FILE, constraints_format='auto',
                 grid_size=9, eps=1e-3, group_order=None, max_rounds=10,
                 cell_order='confidence', group_solver='greedy'):
        super().__init__()
        assert grid_size == 9, 'Only 9x9 Sudoku supported'
        self.grid_size = grid_size
        self.n_cells = grid_size * grid_size
        self.constraints_path = str(constraints_path)
        self.constraints_format = constraints_format

        cs = load_group_constraints(constraints_path, constraints_format)
        validate_group_constraints(cs, grid_size)
        self.kpass = GroupKPassCasper(
            cs, grid_size, max_rounds=max_rounds, eps=eps, cell_order=cell_order,
            group_solver=group_solver,
        )
        self.max_rounds = max_rounds

        canonical = build_groups(grid_size)
        if group_order is None:
            active = list(range(len(canonical)))
        else:
            active = validate_group_order(list(group_order))
        self.group_order = active
        self.kpass.blocks = [canonical[i] for i in active]
        # Non-persistent: the layer adds nothing to checkpoints.
        self.register_buffer(
            'groups',
            torch.tensor([list(c) for _, c in self.kpass.blocks], dtype=torch.long),
            persistent=False,
        )

    def forward(self, cell_probs, max_iterations=None):
        B, N, C = cell_probs.shape
        assert N == self.n_cells and C == self.grid_size
        self.kpass.max_rounds = self.max_rounds if max_iterations is None else int(max_iterations)
        adjusted, info = self.kpass(cell_probs.reshape(B, N * C))
        n_cert = sum(c is not None for c in info['certified_at'])
        info.update(
            avg_l2_per_sample=info['total_l2_distance'] / max(B, 1),
            n_iterations=info['rounds_used'],
            converged=n_cert == B,
            n_certified=n_cert,
            total_constraints_applied=int(sum(sum(r) for r in info['edits_per_round'])),
        )
        return adjusted.view(B, N, C), info

    def constraint_satisfaction_rate(self, cell_probs):
        """Fraction of (sample, group) pairs where all 9 argmax digits are distinct.
        Same definition as SudokuConstraintLayer.constraint_satisfaction_rate."""
        B = cell_probs.shape[0]
        preds = cell_probs.argmax(dim=2)
        n_groups = self.groups.shape[0]
        total = 0
        satisfied = 0
        for g in range(n_groups):
            group_preds = preds[:, self.groups[g]]
            sorted_preds, _ = group_preds.sort(dim=1)
            distinct = (sorted_preds[:, 1:] != sorted_preds[:, :-1]).all(dim=1)
            satisfied += distinct.sum().item()
            total += B
        return satisfied / max(total, 1)


def add_edit_layer_args(parser):
    parser.add_argument('--edit-layer', default='casper', choices=['casper', 'minedit'],
                        help='casper (default): group-local CASPER KPass reading the group '
                             'constraints from --constraints-file. minedit: legacy '
                             'SudokuConstraintLayer (Hungarian / --pairwise).')
    parser.add_argument('--constraints-file', default=str(DEFAULT_CONSTRAINTS_FILE), type=str,
                        help='Compiled CASPER constraint file for one Sudoku group '
                             '(--edit-layer casper only).')
    parser.add_argument('--constraints-format', default='auto',
                        choices=['auto', 'compiled-txt', 'json', 'dsl'],
                        help='Format of --constraints-file. auto: .json or *_compiled.txt.')
    parser.add_argument('--casper-cell-order', default='confidence',
                        choices=['confidence', 'canonical'],
                        help='Within-group visiting order for --edit-layer casper '
                             '(greedy group solver only).')
    parser.add_argument('--casper-group-solver', default='greedy', choices=['greedy', 'search'],
                        help='How --edit-layer casper repairs a group. greedy (default): '
                             'cell by cell, each conflicting cell moves to its nearest free '
                             'digit. search: the lowest -log p complete assignment of the '
                             'group over the compiled feasible sets (branch and bound).')


def build_edit_layer(args, device=None):
    """Single factory for the edit layer used by all pipeline scripts.

    Reads args.edit_layer, args.pairwise, args.constraint_group_order (a parsed
    list or None) and the casper flags from add_edit_layer_args.
    """
    group_order = getattr(args, 'constraint_group_order', None)
    if isinstance(group_order, str):
        group_order = parse_group_order(group_order)
    if args.edit_layer == 'minedit':
        from legacy.sudoku_constraint_layer import SudokuConstraintLayer
        layer = SudokuConstraintLayer(
            grid_size=9, pairwise=getattr(args, 'pairwise', False), group_order=group_order,
        )
        policy = 'pairwise' if getattr(args, 'pairwise', False) else 'joint k-ary AllDifferent'
        print(f'[edit-layer] minedit (policy={policy}, group_order={format_group_order(group_order)})')
    else:
        if getattr(args, 'pairwise', False):
            raise ValueError('--pairwise only applies to --edit-layer minedit')
        layer = CasperSudokuLayer(
            constraints_path=args.constraints_file,
            constraints_format=args.constraints_format,
            group_order=group_order,
            cell_order=args.casper_cell_order,
            group_solver=getattr(args, 'casper_group_solver', 'greedy'),
        )
        print(f'[edit-layer] casper group-local KPass (constraints={args.constraints_file}, '
              f'format={args.constraints_format}, cell_order={args.casper_cell_order}, '
              f'group_solver={getattr(args, "casper_group_solver", "greedy")}, '
              f'group_order={format_group_order(group_order)})')
    return layer.to(device) if device is not None else layer
