"""Extract cross-group clauses from a compiled sudoku constraint file.

A "cross-group" clause is one whose variables do NOT all lie within a single
row, column, or box of the sudoku grid — i.e. genuinely derived constraints
that the resolution compile produced beyond the original input pairings.

Usage:
    python3 examples/extract_cross_group.py <compiled.txt> <out.txt> [--grid 4]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "casper"))

from signed_clauses import SignedClause, parse_constraints  # noqa: E402


def sudoku_groups(grid: int) -> list[set[int]]:
    """Row, column, and (for perfect-square grids) box groups for an N×N sudoku.

    Cells are labelled y_1..y_{grid*grid}, row-major. Returns each group as
    a set of label numbers (1-indexed)."""
    label = lambda i, j: i * grid + j + 1
    rows = [{label(i, j) for j in range(grid)} for i in range(grid)]
    cols = [{label(i, j) for i in range(grid)} for j in range(grid)]
    groups = rows + cols
    box_side = int(grid ** 0.5)
    if box_side * box_side == grid:
        for bi in range(box_side):
            for bj in range(box_side):
                groups.append({
                    label(bi * box_side + di, bj * box_side + dj)
                    for di in range(box_side) for dj in range(box_side)
                })
    return groups


def fmt_clause(clause: SignedClause, var_names: list[str], value_base: int) -> str:
    parts = []
    for lit in clause.literals:
        vs = ",".join(str(v + value_base) for v in sorted(lit.values))
        parts.append(f"[{vs}]:{var_names[lit.var]}")
    return " or ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="Compiled .txt to read")
    p.add_argument("output", type=Path, help="Where to write cross-group clauses")
    p.add_argument("--grid", type=int, default=4, help="Sudoku grid size (default 4)")
    args = p.parse_args()

    cs = parse_constraints(args.input.read_text())
    groups = sudoku_groups(args.grid)

    label = {
        i: int(re.match(r"y_(\d+)", n).group(1)) for i, n in enumerate(cs.var_names)
    }

    lines: list[str] = [
        "# Cross-group clauses extracted from compiled constraints.",
        f"# Source: {args.input}",
        f"# Sudoku grid: {args.grid}x{args.grid} (row + col + box groups)",
        f"# Values displayed using value-base {cs.value_base}.",
        "# A clause is cross-group iff its variables don't all lie in a single",
        "# row, column, or box.",
        "",
    ]

    by_length: dict[int, list[SignedClause]] = {}
    structural = 0
    for c in cs.clauses:
        vs = frozenset(label[lit.var] for lit in c.literals)
        k = len(vs)
        if any(vs.issubset(g) for g in groups):
            structural += 1
        else:
            by_length.setdefault(k, []).append(c)

    total_cross = sum(len(v) for v in by_length.values())

    lines.append(
        f"# Total clauses in source: {len(cs.clauses)} "
        f"(structural={structural}, cross-group={total_cross})"
    )
    lines.append("")

    for k in sorted(by_length):
        bucket = by_length[k]
        lines.append(
            f"# ============ {k}-literal cross-group clauses ({len(bucket)}) ============"
        )
        for c in bucket:
            lines.append(fmt_clause(c, cs.var_names, cs.value_base))
        lines.append("")

    args.output.write_text("\n".join(lines) + "\n")
    print(
        f"Wrote {total_cross} cross-group clauses "
        f"(of {len(cs.clauses)} total; {structural} structural) -> {args.output}"
    )


if __name__ == "__main__":
    main()
