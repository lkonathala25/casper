from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "casper"))

from casper_layer import CasperLayer
from constraints import Constraints


ROWS = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)]
COLS = [(0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15)]
BOXES = [(0, 1, 4, 5), (2, 3, 6, 7), (8, 9, 12, 13), (10, 11, 14, 15)]

N_VARS = 16
DOMAIN = 4

VALID_GRID: list[int] = [0, 1, 2, 3, 2, 3, 0, 1, 1, 0, 3, 2, 3, 2, 1, 0]


def verify_sudoku_4x4(grid: list[int]) -> tuple[bool, list[str]]:
    """Return (ok, violations). `grid` is 16 ints in row-major order."""
    bad: list[str] = []
    target = [0, 1, 2, 3]
    for kind, groups in (("row", ROWS), ("col", COLS), ("box", BOXES)):
        for i, g in enumerate(groups):
            vals = sorted(grid[k] for k in g)
            if vals != target:
                bad.append(f"{kind} {i}: {[grid[k] for k in g]}")
    return (not bad), bad


def render_grid(grid: list[int]) -> str:
    """4×4 grid pretty-printed with 1-based values and box separators."""
    lines: list[str] = []
    for r in range(4):
        cells = []
        for c in range(4):
            cells.append(str(grid[r * 4 + c] + 1))
            if c == 1:
                cells.append("|")
        lines.append(" ".join(cells))
        if r == 1:
            lines.append("-" * 9)
    return "\n".join("    " + ln for ln in lines)


def softmax_per_var(logits: torch.Tensor) -> torch.Tensor:
    """Apply softmax over each variable's domain slice; preserves the autograd graph."""
    B = logits.shape[0]
    return torch.softmax(logits.view(B, N_VARS, DOMAIN), dim=-1).view(B, -1)


def extract_assignments_from_probs(
    probs: torch.Tensor, n_vars: int = N_VARS, domain: int = DOMAIN
) -> list[list[int]]:
    """Per-sample argmax over each variable's domain slice."""
    B = probs.shape[0]
    return [
        [int(probs[b, v * domain : (v + 1) * domain].argmax().item()) for v in range(n_vars)]
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


def run_layer(
    layer: CasperLayer, probs: torch.Tensor
) -> tuple[list[list[int]], list[list[int]], dict, int, int, int]:
    """Run probs through layer and return (pre_grids, post_grids, info,
    n_valid_pre, n_valid_post, cells_edited)."""
    pre = extract_assignments_from_probs(probs)
    adjusted, info = layer(probs)
    post = extract_assignments_from_probs(adjusted)
    B = probs.shape[0]
    n_valid_pre = sum(verify_sudoku_4x4(g)[0] for g in pre)
    n_valid_post = sum(verify_sudoku_4x4(g)[0] for g in post)
    cells_edited = sum(
        int(pre[b][k] != post[b][k]) for b in range(B) for k in range(N_VARS)
    )
    return pre, post, info, n_valid_pre, n_valid_post, cells_edited


def report(name: str, info: dict, B: int, n_valid_pre: int, n_valid_post: int, cells_edited: int) -> None:
    """One-line summary of a layer pass."""
    print(
        f"  [{name}] B={B}  valid: {n_valid_pre}->{n_valid_post}  "
        f"cells edited: {cells_edited}  "
        f"total_l2: {info['total_l2_distance']:.4f}  "
        f"avg_l2/sample: {info['avg_l2_per_sample']:.4f}"
    )


def test_random_batch(layer: CasperLayer) -> bool:
    """Random softmax inputs should always project to a valid grid; print per-sample detail."""
    print("\nTest 1: random softmax batch (B=8) — detail view")
    torch.manual_seed(0)
    logits = torch.randn(8, N_VARS * DOMAIN)
    probs = softmax_per_var(logits)

    t0 = time.time()
    pre, post, info, n_valid_pre, n_valid_post, cells_edited = run_layer(layer, probs)
    t_forward = time.time() - t0

    print(
        f"  forward pass: {t_forward:.3f}s ({t_forward / 8 * 1000:.1f} ms/sample)"
    )
    report("random", info, 8, n_valid_pre, n_valid_post, cells_edited)

    for b in (0, 7):  # show first and last as a sanity peek
        ok_post, bad = verify_sudoku_4x4(post[b])
        print(f"\n  Sample {b} pre (argmax):")
        print(render_grid(pre[b]))
        print(f"  Sample {b} post (Casper-projected) [{'VALID' if ok_post else 'INVALID'}]:")
        print(render_grid(post[b]))

    ok = n_valid_post == 8
    print(f"\n  -> {'PASS' if ok else 'FAIL'}: expect 8/8 valid post-projection")
    return ok


def test_already_valid(layer: CasperLayer) -> bool:
    """Sharply-peaked valid grid should be a no-op: 0 edits, ~0 L2."""
    print("\nTest 2: already-valid sharply-peaked input (peak=0.97)")
    probs = peaked_probs([VALID_GRID], peak=0.97)
    _, _, info, n_valid_pre, n_valid_post, cells_edited = run_layer(layer, probs)
    report("already_valid", info, 1, n_valid_pre, n_valid_post, cells_edited)
    ok = (
        n_valid_post == 1
        and cells_edited == 0
        and info["total_l2_distance"] < 1e-3
    )
    print(
        f"  -> {'PASS' if ok else 'FAIL'}: "
        f"expect cells_edited=0 and total_l2 < 1e-3"
    )
    return ok


def test_near_valid(layer: CasperLayer) -> bool:
    """1- and 2-cell perturbations should project to a valid grid with L2 << random."""
    print("\nTest 3: near-valid input (1- and 2-cell perturbations)")
    grids = [
        flip_cells(VALID_GRID, [0]),
        flip_cells(VALID_GRID, [0, 5]),
    ]
    probs = peaked_probs(grids, peak=0.95)
    _, post, info, n_valid_pre, n_valid_post, cells_edited = run_layer(layer, probs)
    report("near_valid", info, 2, n_valid_pre, n_valid_post, cells_edited)

    torch.manual_seed(1)
    rand_probs = softmax_per_var(torch.randn(2, N_VARS * DOMAIN))
    _, _, rand_info, _, _, _ = run_layer(layer, rand_probs)
    print(
        f"    baseline random avg_l2/sample = {rand_info['avg_l2_per_sample']:.4f}  "
        f"vs near-valid avg_l2/sample = {info['avg_l2_per_sample']:.4f}"
    )

    ok = (
        n_valid_post == 2
        and info["avg_l2_per_sample"] < rand_info["avg_l2_per_sample"]
    )
    print(
        f"  -> {'PASS' if ok else 'FAIL'}: "
        f"expect 2/2 valid and near-valid L2 strictly < random L2"
    )
    return ok


def test_uniform(layer: CasperLayer) -> bool:
    """Max-entropy uniform input should still produce a valid grid."""
    print("\nTest 4: max-entropy uniform probabilities (all 0.25)")
    probs = torch.full((1, N_VARS * DOMAIN), 1.0 / DOMAIN)
    _, post, info, n_valid_pre, n_valid_post, cells_edited = run_layer(layer, probs)
    report("uniform", info, 1, n_valid_pre, n_valid_post, cells_edited)
    ok = n_valid_post == 1
    print(f"  -> {'PASS' if ok else 'FAIL'}: expect 1/1 valid (no errors, valid grid emitted)")
    if ok:
        print("  Uniform-input grid produced:")
        print(render_grid(post[0]))
    return ok


def test_gradient_flow(layer: CasperLayer) -> bool:
    """projection_loss.backward() should produce a finite, non-zero gradient on input logits."""
    print("\nTest 5: gradient flow through the layer")
    torch.manual_seed(2)
    logits = torch.randn(4, N_VARS * DOMAIN, requires_grad=True)
    probs = softmax_per_var(logits)
    adjusted, info = layer(probs)
    loss = info["projection_loss"]
    loss.backward()

    has_grad = logits.grad is not None
    finite = bool(torch.isfinite(logits.grad).all().item()) if has_grad else False
    nonzero = bool((logits.grad.abs().max() > 0).item()) if has_grad else False
    max_abs = float(logits.grad.abs().max().item()) if has_grad else float("nan")

    print(
        f"    projection_loss={float(loss.item()):.4f}  "
        f"grad shape={tuple(logits.grad.shape) if has_grad else None}  "
        f"max|grad|={max_abs:.4f}  finite={finite}  nonzero={nonzero}"
    )
    ok = has_grad and finite and nonzero
    print(f"  -> {'PASS' if ok else 'FAIL'}: expect non-zero finite gradient on input logits")
    return ok


def main() -> None:
    compiled_path = ROOT / "constraints" / "sudoku" / "sudoku_4x4_compiled.txt"

    t0 = time.time()
    print(f"Loading {compiled_path.name} (no recompile)...")
    constraints = Constraints.from_compiled_dsl(compiled_path)
    t_load = time.time() - t0
    print(f"  loaded in {t_load:.2f}s; n_vars={len(constraints.var_domains)}")

    layer = CasperLayer(constraints)
    print(
        f"  CasperLayer built; ordering_levels lengths = "
        f"{[len(lvl) for lvl in constraints.ordering_levels]}"
    )

    results = {
        "random_batch": test_random_batch(layer),
        "already_valid": test_already_valid(layer),
        "near_valid": test_near_valid(layer),
        "uniform": test_uniform(layer),
        "gradient_flow": test_gradient_flow(layer),
    }

    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    n_pass = sum(results.values())
    print(f"  Total: {n_pass}/{len(results)} passed")


if __name__ == "__main__":
    main()
