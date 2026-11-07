"""Shared parsing/validation for the configurable 27-group Sudoku constraint order.

Canonical indices match SudokuConstraintLayer._build_groups(): 0..8 rows,
9..17 columns, 18..26 boxes. Centralized here (rather than duplicated per
entrypoint) so rl_eval_sudoku.py, evaluate_checkpoints.py, minedit_sudoku.py,
minedit_mask_calibrate.py and minedit_llm_mask_calibrate.py all share one
implementation, and stays import-light (no torch) so the orchestrator script
(evaluate_checkpoints.py) doesn't need to import torch just to validate a CLI
string.
"""

N_GROUPS = 27


def validate_group_order(order):
    """Raise ValueError unless `order` is a length-27 permutation of 0..26.

    Returns the input as a plain list[int] on success.
    """
    order = list(order)
    if len(order) != N_GROUPS:
        raise ValueError(
            f'--constraint-group-order must have exactly {N_GROUPS} entries, '
            f'got {len(order)}: {order}'
        )
    for v in order:
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(
                f'--constraint-group-order entries must be integers, got {v!r}'
            )

    seen = set()
    dups = set()
    out_of_range = set()
    for v in order:
        if v < 0 or v >= N_GROUPS:
            out_of_range.add(v)
        if v in seen:
            dups.add(v)
        seen.add(v)
    missing = sorted(set(range(N_GROUPS)) - seen)

    if dups or out_of_range or missing:
        raise ValueError(
            'Invalid --constraint-group-order: must be a permutation of '
            f'0..{N_GROUPS - 1}. duplicates={sorted(dups)} '
            f'out_of_range={sorted(out_of_range)} missing={missing} '
            f'(got: {order})'
        )
    return order


def parse_group_order(order_str):
    """Parse a CLI string into a validated list[int] permutation, or None for default.

    None / '' / 'default' (case-insensitive) -> None (keep canonical order).
    Otherwise: comma-separated integers, validated as a 0..26 permutation.
    Raises ValueError loudly on any malformed or invalid input.
    """
    if order_str is None:
        return None
    s = str(order_str).strip()
    if s == '' or s.lower() == 'default':
        return None

    tokens = s.split(',')
    values = []
    for tok in tokens:
        tok = tok.strip()
        try:
            values.append(int(tok))
        except ValueError:
            raise ValueError(
                f"Invalid --constraint-group-order: non-integer token '{tok}' "
                f"in '{order_str}'"
            )
    return validate_group_order(values)


def format_group_order(order):
    """Render an order (or None for canonical/default) as a compact string."""
    if order is None:
        return 'default(0..26)'
    return ','.join(str(x) for x in order)
