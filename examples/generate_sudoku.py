from __future__ import annotations

import argparse
import math
from pathlib import Path


def is_perfect_square(n: int) -> int | None:
    s = int(math.isqrt(n))
    return s if s * s == n else None


def generate_sudoku_dsl(grid_size: int) -> str:
    """Emit a signed-CNF DSL string encoding an N×N Sudoku/Latin square.

    For grid_size = box_side^2 (4, 9, 16, ...), box constraints are added on
    top of row + column ALLDIFFERENTs. For other sizes only row + column
    constraints are emitted (Latin square).

    Emits both at-most-one (AMO) and at-least-one (ALO) clauses for each
    group — together this is the standard exactly-one ALLDIFFERENT encoding.
    AMO alone is provably hard for resolution at h ≥ 4 (pigeonhole barrier);
    ALO is what lets the compile escape that barrier.
    """
    g = grid_size
    box_side = is_perfect_square(g)

    cells = [(i, j) for i in range(g) for j in range(g)]
    cell_index = {c: k + 1 for k, c in enumerate(cells)}
    name = lambda c: f"y_{cell_index[c]}"

    lines: list[str] = []
    for c in cells:
        lines.append(f"var {name(c)} {g}")
    lines.append(f"ordering " + " ".join(name(c) for c in cells))
    lines.append("")

    groups: list[list] = []
    for i in range(g):
        groups.append([(i, j) for j in range(g)])
    for j in range(g):
        groups.append([(i, j) for i in range(g)])
    if box_side is not None:
        for bi in range(box_side):
            for bj in range(box_side):
                groups.append([
                    (bi * box_side + di, bj * box_side + dj)
                    for di in range(box_side)
                    for dj in range(box_side)
                ])

    pairs: set[frozenset] = set()
    for group in groups:
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                pairs.add(frozenset({group[a], group[b]}))

    for pair in sorted(pairs, key=lambda p: tuple(sorted(p))):
        a, b = sorted(pair)
        for v in range(g):
            others = ",".join(str(x) for x in range(g) if x != v)
            lines.append(f"[{others}]:{name(a)} or [{others}]:{name(b)}")

    for group in groups:
        for v in range(g):
            parts = [f"[{v}]:{name(c)}" for c in group]
            lines.append(" or ".join(parts))

    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a Sudoku/Latin-square constraint file.")
    p.add_argument("grid_size", type=int)
    p.add_argument("out_path", type=Path)
    args = p.parse_args()

    dsl = generate_sudoku_dsl(args.grid_size)
    args.out_path.write_text(dsl)
    n_clauses = sum(
        1 for line in dsl.splitlines()
        if line and not line.startswith(("var", "ordering", "#"))
    )
    print(f"Wrote {n_clauses} clauses to {args.out_path}")


if __name__ == "__main__":
    main()
