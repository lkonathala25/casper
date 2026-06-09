from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from casper_layer import CasperLayer
from constraints import Constraints


ROWS = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)]
COLS = [(0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15)]
BOXES = [(0, 1, 4, 5), (2, 3, 6, 7), (8, 9, 12, 13), (10, 11, 14, 15)]

N_VARS = 16
DOMAIN = 4

VALID_GRID: list[int] = [0, 1, 2, 3, 2, 3, 0, 1, 1, 0, 3, 2, 3, 2, 1, 0]


def conflict_mask(grid: list[int]) -> list[bool]:
    """Mark cells whose value duplicates another in the same row, col, or box."""
    bad = [False] * 16
    for groups in (ROWS, COLS, BOXES):
        for g in groups:
            counts: dict[int, int] = {}
            for k in g:
                counts[grid[k]] = counts.get(grid[k], 0) + 1
            for k in g:
                if counts[grid[k]] > 1:
                    bad[k] = True
    return bad


def softmax_per_var(logits: torch.Tensor) -> torch.Tensor:
    """Softmax over each variable's domain slice; preserves autograd graph."""
    B = logits.shape[0]
    return torch.softmax(logits.view(B, N_VARS, DOMAIN), dim=-1).view(B, -1)


def extract_grids(probs: torch.Tensor) -> list[list[int]]:
    """Per-sample argmax over each variable's domain slice."""
    B = probs.shape[0]
    return [
        [int(probs[b, v * DOMAIN : (v + 1) * DOMAIN].argmax().item()) for v in range(N_VARS)]
        for b in range(B)
    ]


def peaked_probs(grids: list[list[int]], peak: float = 0.97) -> torch.Tensor:
    """Build sharply-peaked softmax probs whose argmax matches each given grid."""
    B = len(grids)
    other = (1.0 - peak) / (DOMAIN - 1)
    probs = torch.full((B, N_VARS, DOMAIN), other)
    for b, grid in enumerate(grids):
        for v, val in enumerate(grid):
            probs[b, v, val] = peak
    return probs.view(B, -1)


def flip_cells(grid: list[int], indices: list[int]) -> list[int]:
    """Return a copy of `grid` with each cell at `indices` shifted by +1 mod DOMAIN."""
    out = list(grid)
    for idx in indices:
        out[idx] = (out[idx] + 1) % DOMAIN
    return out


def draw_grid(
    ax,
    values: list[int],
    hl_mask: list[bool],
    title: str,
    hl_face: str,
    hl_edge: str,
    hl_text: str,
) -> None:
    """Render a 4×4 sudoku grid with highlights for `hl_mask` cells."""
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, pad=4)
    for r in range(4):
        for c in range(4):
            idx = r * 4 + c
            face = hl_face if hl_mask[idx] else "#ffffff"
            edge = hl_edge if hl_mask[idx] else "#cccccc"
            ax.add_patch(
                patches.Rectangle(
                    (c, r), 1, 1, facecolor=face, edgecolor=edge, linewidth=1.0
                )
            )
            v = values[idx] + 1
            color = hl_text if hl_mask[idx] else "#222222"
            ax.text(
                c + 0.5,
                r + 0.5,
                str(v),
                ha="center",
                va="center",
                fontsize=18,
                color=color,
                fontweight="bold",
            )
    for k in (0, 2, 4):
        ax.plot([k, k], [0, 4], color="black", linewidth=2.0)
        ax.plot([0, 4], [k, k], color="black", linewidth=2.0)


def build_cases(layer: CasperLayer) -> list[dict]:
    """Build the list of cases to visualize. Each case has 'label', 'subtitle',
    'pre', 'post', 'pre_mask', 'post_mask'."""
    cases: list[dict] = []

    torch.manual_seed(0)
    rand_logits = torch.randn(2, N_VARS * DOMAIN)
    rand_probs = softmax_per_var(rand_logits)
    rand_pre = extract_grids(rand_probs)
    rand_adj, rand_info = layer(rand_probs)
    rand_post = extract_grids(rand_adj)
    for i in (0, 1):
        cases.append(
            {
                "label": f"Random softmax (sample {i})",
                "subtitle": f"L2 = {rand_info['avg_l2_per_sample']:.2f}",
                "pre": rand_pre[i],
                "post": rand_post[i],
                "pre_mask": conflict_mask(rand_pre[i]),
                "post_mask": [rand_pre[i][k] != rand_post[i][k] for k in range(16)],
            }
        )

    vp = peaked_probs([VALID_GRID], peak=0.97)
    vp_pre = extract_grids(vp)
    vp_adj, vp_info = layer(vp)
    vp_post = extract_grids(vp_adj)
    cases.append(
        {
            "label": "Already-valid (peak=0.97)",
            "subtitle": f"L2 = {vp_info['total_l2_distance']:.4f}  (expected ~0)",
            "pre": vp_pre[0],
            "post": vp_post[0],
            "pre_mask": [False] * 16,
            "post_mask": [vp_pre[0][k] != vp_post[0][k] for k in range(16)],
        }
    )

    nv_grids = [flip_cells(VALID_GRID, [0]), flip_cells(VALID_GRID, [0, 5])]
    nv = peaked_probs(nv_grids, peak=0.95)
    nv_pre = extract_grids(nv)
    nv_adj, nv_info = layer(nv)
    nv_post = extract_grids(nv_adj)
    for i, n_flips in enumerate([1, 2]):
        cases.append(
            {
                "label": f"Near-valid ({n_flips} cell{'s' if n_flips > 1 else ''} flipped)",
                "subtitle": f"L2 = {(nv_info['total_l2_distance'] / 2):.4f}",
                "pre": nv_pre[i],
                "post": nv_post[i],
                "pre_mask": conflict_mask(nv_pre[i]),
                "post_mask": [nv_pre[i][k] != nv_post[i][k] for k in range(16)],
            }
        )

    up = torch.full((1, N_VARS * DOMAIN), 1.0 / DOMAIN)
    up_pre = extract_grids(up)
    up_adj, up_info = layer(up)
    up_post = extract_grids(up_adj)
    cases.append(
        {
            "label": "Max-entropy uniform (0.25 everywhere)",
            "subtitle": f"L2 = {up_info['total_l2_distance']:.4f}",
            "pre": up_pre[0],
            "post": up_post[0],
            "pre_mask": conflict_mask(up_pre[0]),
            "post_mask": [up_pre[0][k] != up_post[0][k] for k in range(16)],
        }
    )

    return cases


def main() -> None:
    compiled_path = ROOT / "examples" / "sudoku_4x4_compiled.txt"
    constraints = Constraints.from_compiled_dsl(compiled_path)
    layer = CasperLayer(constraints)

    cases = build_cases(layer)
    n = len(cases)

    fig, axes = plt.subplots(
        n, 2, figsize=(9.5, 3.2 * n), gridspec_kw={"wspace": 0.35, "hspace": 0.65}
    )
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, case in enumerate(cases):
        n_conf = sum(case["pre_mask"])
        n_moved = sum(case["post_mask"])
        draw_grid(
            axes[i, 0],
            case["pre"],
            case["pre_mask"],
            title=f"{case['label']}: pre\n{n_conf} cells in conflict",
            hl_face="#fde2e2",
            hl_edge="#cc3333",
            hl_text="#a40000",
        )
        draw_grid(
            axes[i, 1],
            case["post"],
            case["post_mask"],
            title=f"{case['label']}: post Casper\n{n_moved} cells moved   {case['subtitle']}",
            hl_face="#dff0d8",
            hl_edge="#2e7d32",
            hl_text="#1b5e20",
        )
        axes[i, 0].annotate(
            "",
            xy=(1.18, 0.5),
            xytext=(1.02, 0.5),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=2),
        )

    fig.suptitle(
        "CasperLayer on 4x4 Sudoku — random + edge cases\n"
        "left: input argmax (red = constraint violation)    "
        "right: post-projection (green = cell edited by the layer)",
        fontsize=12,
        y=0.995,
    )
    fig.subplots_adjust(top=0.96, bottom=0.02, left=0.05, right=0.97)

    out_path = ROOT / "examples" / "sudoku_4x4_layer_demo.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
