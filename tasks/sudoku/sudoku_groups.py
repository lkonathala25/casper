from __future__ import annotations

import math


def box_id(r: int, c: int, box_side: int) -> int:
    """0-based box index of cell (r, c) on a grid with boxes of side `box_side`."""
    return (r // box_side) * box_side + (c // box_side)


def cell_index(r: int, c: int, g: int) -> int:
    """0-based row-major variable index of cell (r, c) on a g x g grid."""
    return r * g + c


def build_groups(g: int = 9) -> list[tuple[str, tuple[int, ...]]]:
    """The 3g ALLDIFFERENT groups of a g x g Sudoku, as (name, cell indices).

    Ordered rows, then columns, then boxes. Cell indices are 0-based row-major.
    The order matters at inference time: groups are processed in this sequence
    and each pass overwrites whatever earlier passes wrote, so the group that
    touches a cell last is the one whose constraint that cell ends up
    satisfying.
    """
    box_side = int(math.isqrt(g))
    groups: list[tuple[str, tuple[int, ...]]] = []

    for r in range(g):
        groups.append((f"row{r + 1}", tuple(cell_index(r, c, g) for c in range(g))))
    for c in range(g):
        groups.append((f"col{c + 1}", tuple(cell_index(r, c, g) for r in range(g))))

    if box_side * box_side == g:
        for b in range(g):
            br, bc = divmod(b, box_side)
            groups.append((
                f"box{b + 1}",
                tuple(
                    cell_index(br * box_side + dr, bc * box_side + dc, g)
                    for dr in range(box_side)
                    for dc in range(box_side)
                ),
            ))
    return groups


def var_to_groups(g: int = 9) -> dict[int, tuple[str, ...]]:
    """Cell index -> names of the groups containing it (its row, column, box)."""
    table: dict[int, list[str]] = {v: [] for v in range(g * g)}
    for name, cells in build_groups(g):
        for v in cells:
            table[v].append(name)
    return {v: tuple(names) for v, names in table.items()}


def group_dsl(size: int = 9, value_base: int = 1) -> str:
    """Signed-clause DSL for one ALLDIFFERENT group of `size` mutually-distinct
    variables over `size` values.

    Emits the same two clause families as the full-board generator: pairwise
    "not both v" clauses for every pair, and a covering clause per value. The
    variables are the group-local y_1 .. y_size, so a single compile of this
    file is reusable for every group on the board under a relabelling of local
    slot to board cell.
    """
    disp = lambda x: str(x + value_base)
    names = [f"y_{i + 1}" for i in range(size)]

    lines = [f"var {n} {size}" for n in names]
    if value_base:
        lines += ["", f"value-base {value_base}"]
    lines += ["", "ordering " + " ".join(names), ""]

    for a in range(size):
        for b in range(a + 1, size):
            for v in range(size):
                others = ",".join(disp(x) for x in range(size) if x != v)
                lines.append(f"[{others}]:{names[a]} or [{others}]:{names[b]}")

    for v in range(size):
        lines.append(" or ".join(f"[{disp(v)}]:{n}" for n in names))

    return "\n".join(lines) + "\n"
