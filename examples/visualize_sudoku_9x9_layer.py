from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
for _sub in ("casper", "tasks/sudoku"):
    if str(ROOT / _sub) not in sys.path:
        sys.path.insert(0, str(ROOT / _sub))

from constraints import Constraints
from local_casper import (
    LocalGroupCasper,
    alldiff_group_constraints,
    extract_assignments,
)
from sudoku_groups import build_groups

N = 9
N_CELLS = 81
BOX = 3
ARTIFACT = ROOT / "constraints" / "sudoku" / "sudoku_group9_compiled.json"

GROUPS = build_groups(N)
GROUP_BY_NAME = dict(GROUPS)

VALID_BOARD = [(3 * (r % 3) + r // 3 + c) % 9 for r in range(N) for c in range(N)]


def conflict_mask(board: list[int]) -> list[bool]:
    """Mark cells whose digit duplicates another inside one of its groups."""
    bad = [False] * N_CELLS
    for _name, cells in GROUPS:
        counts: dict[int, int] = {}
        for k in cells:
            counts[board[k]] = counts.get(board[k], 0) + 1
        for k in cells:
            if counts[board[k]] > 1:
                bad[k] = True
    return bad


def violations_by_type(board: list[int]) -> dict[str, int]:
    """Count violated groups split by row / col / box."""
    out = {"rows": 0, "cols": 0, "boxes": 0}
    key = {"r": "rows", "c": "cols", "b": "boxes"}
    for name, cells in GROUPS:
        if sorted(board[k] for k in cells) != list(range(N)):
            out[key[name[0]]] += 1
    return out


def clean_probs(board: list[int], peak: float = 0.80) -> torch.Tensor:
    """(1, 729) scores: `peak` on each cell's intended digit, rest uniform."""
    other = (1.0 - peak) / (N - 1)
    t = torch.full((1, N_CELLS, N), other, dtype=torch.float64)
    for cell, val in enumerate(board):
        t[0, cell, val] = peak
    return t.reshape(1, N_CELLS * N)


def corrupt(probs: torch.Tensor, cell: int, wrong: int, true: int) -> torch.Tensor:
    """0.80 on the wrong digit, 0.15 on the truth, 0.05/7 over the rest."""
    out = probs.clone().view(1, N_CELLS, N)
    out[0, cell, :] = 0.05 / (N - 2)
    out[0, cell, wrong] = 0.80
    out[0, cell, true] = 0.15
    return out.reshape(1, N_CELLS * N)


def row_clash_digit(cell: int) -> int:
    """A digit already held earlier in this cell's row — a guaranteed duplicate."""
    r = cell // N
    first = GROUP_BY_NAME[f"row{r + 1}"][0]
    return VALID_BOARD[first]


def draw_grid(ax, values, face_mask, edge_mask, title) -> None:
    """Render a 9x9 board; `face_mask` fills red, `edge_mask` outlines green."""
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9, pad=5)
    for r in range(N):
        for c in range(N):
            idx = r * N + c
            red, green = face_mask[idx], edge_mask[idx]
            if red:
                face, edge, txt = "#fde2e2", "#cc3333", "#a40000"
            elif green:
                face, edge, txt = "#dff0d8", "#2e7d32", "#1b5e20"
            else:
                face, edge, txt = "#ffffff", "#cccccc", "#222222"
            lw = 1.6 if (red and green) else 1.0
            if red and green:
                edge = "#2e7d32"
            ax.add_patch(
                patches.Rectangle(
                    (c, r), 1, 1, facecolor=face, edgecolor=edge, linewidth=lw
                )
            )
            ax.text(
                c + 0.5, r + 0.5, str(values[idx] + 1),
                ha="center", va="center", fontsize=9,
                color=txt, fontweight="bold",
            )
    for k in range(0, N + 1, BOX):
        ax.plot([k, k], [0, N], color="black", linewidth=2.0)
        ax.plot([0, N], [k, k], color="black", linewidth=2.0)


def run_case(compiled, closed, probs, label):
    """Run one input through both paths, assert equality, return the panel data."""
    with torch.no_grad():
        out_c, info_c = compiled(probs)
        out_f, _info_f = closed(probs)
    same = torch.equal(out_c, out_f)
    if not same:
        raise SystemExit(
            f"HALT: compiled != from_function tensor on case {label!r}"
        )
    pre = extract_assignments(probs, N_CELLS, N)[0]
    post = extract_assignments(out_c, N_CELLS, N)[0]
    moved = [k for k in range(N_CELLS) if pre[k] != post[k]]
    post_conf = conflict_mask(post)
    first_edit = (
        f"pass {info_c['edits_by_pass'][0]['pass']} "
        f"({info_c['edits_by_pass'][0]['group']})"
        if info_c["edits_by_pass"] else "none"
    )
    return {
        "label": label,
        "pre": pre,
        "post": post,
        "pre_mask": conflict_mask(pre),
        "post_red": post_conf,
        "post_green": [pre[k] != post[k] for k in range(N_CELLS)],
        "moved": moved,
        "resid": sum(violations_by_type(post).values()),
        "resid_by_type": violations_by_type(post),
        "l2": float(info_c["total_l2_distance"]),
        "first_edit": first_edit,
        "paths_equal": same,
    }


def build_cases(compiled, closed) -> list[dict]:
    """The six panels, all built on the same known-valid base board."""
    cases = []

    cases.append(run_case(
        compiled, closed, clean_probs(VALID_BOARD), "1. Already-valid (peak 0.80)"
    ))

    # cell 1 sits at row position 1, so its row pass sees the clash immediately
    c1 = 1
    cases.append(run_case(
        compiled, closed,
        corrupt(clean_probs(VALID_BOARD), c1, row_clash_digit(c1), VALID_BOARD[c1]),
        "2. 1 constrained cell corrupted",
    ))

    # cell 9 is first in its row; the row pass has an empty prior and cannot
    # reject the wrong digit, so the error propagates instead of being repaired
    c2 = 9
    wrong2 = VALID_BOARD[0]
    cases.append(run_case(
        compiled, closed,
        corrupt(clean_probs(VALID_BOARD), c2, wrong2, VALID_BOARD[c2]),
        "3. 1 position-0 cell corrupted (known-hard)",
    ))

    p = clean_probs(VALID_BOARD)
    for cell in (7, 40, 77):
        p = corrupt(p, cell, row_clash_digit(cell), VALID_BOARD[cell])
    cases.append(run_case(compiled, closed, p, "4. 3 constrained cells corrupted"))

    torch.manual_seed(0)
    logits = torch.randn(1, N_CELLS, N, dtype=torch.float64)
    rand = torch.softmax(logits, dim=-1).reshape(1, N_CELLS * N)
    cases.append(run_case(compiled, closed, rand, "5. Random softmax"))

    uni = torch.full((1, N_CELLS * N), 1.0 / N, dtype=torch.float64)
    cases.append(run_case(compiled, closed, uni, "6. Max-entropy uniform (1/9)"))

    return cases


def main() -> None:
    compiled = LocalGroupCasper(Constraints.from_file(ARTIFACT), GROUPS, N)
    closed = LocalGroupCasper(alldiff_group_constraints(N), GROUPS, N)

    cases = build_cases(compiled, closed)
    n = len(cases)

    fig, axes = plt.subplots(
        n, 2, figsize=(11, 4.6 * n), gridspec_kw={"wspace": 0.30, "hspace": 0.42}
    )
    for i, case in enumerate(cases):
        draw_grid(
            axes[i, 0], case["pre"], case["pre_mask"], [False] * N_CELLS,
            f"{case['label']}: pre\n{sum(case['pre_mask'])} cells in conflict",
        )
        draw_grid(
            axes[i, 1], case["post"], case["post_red"], case["post_green"],
            f"{case['label']}: post\n{len(case['moved'])} cells moved   "
            f"{case['resid']} residual violations   L2 = {case['l2']:.3f}",
        )
        axes[i, 0].annotate(
            "", xy=(1.16, 0.5), xytext=(1.02, 0.5), xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=2),
        )

    fig.suptitle(
        "Per-group (local-only) CasperLayer on 9x9 Sudoku — 27 sequential passes "
        "(rows 1-9, cols 1-9, boxes 1-9)\n"
        "left: input argmax (red = cell in a violated group)     "
        "right: post-projection (green = edited by the layer, "
        "red = STILL in a violated group)\n"
        "Residual red on the right is expected: per-group projection gives local, "
        "not global, consistency.",
        fontsize=11, y=0.998,
    )
    fig.subplots_adjust(top=0.965, bottom=0.01, left=0.04, right=0.97)

    out = ROOT / "examples" / "sudoku_9x9_layer_demo.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"saved {out}\n")
    for c in cases:
        t = c["resid_by_type"]
        print(f"{c['label']}")
        print(f"    paths equal (compiled == from_function): {c['paths_equal']}")
        print(f"    pre conflicts: {sum(c['pre_mask'])} cells")
        print(f"    cells moved  : {len(c['moved'])} {c['moved'] if len(c['moved']) <= 12 else ''}")
        print(f"    residual violations: {c['resid']} "
              f"(rows {t['rows']}, cols {t['cols']}, boxes {t['boxes']})")
        print(f"    first edit   : {c['first_edit']}")
        print(f"    L2           : {c['l2']:.4f}\n")


if __name__ == "__main__":
    main()
