from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
for _sub in ("casper", "tasks/sudoku"):
    if str(ROOT / _sub) not in sys.path:
        sys.path.insert(0, str(ROOT / _sub))

from casper_layer import CasperLayer
from constraints import Constraints
from generate_sudoku import generate_sudoku_dsl
from local_casper import KPassCasper, extract_assignments, group_violations
from resolution import _heuristic_elimination_ordering, compile_constraints, to_dsl
from signed_clauses import parse_constraints
from sudoku_groups import build_groups

SUDOKU_DIR = ROOT / "constraints" / "sudoku"
COMPILED_4X4 = SUDOKU_DIR / "sudoku_4x4_compiled.txt"
COMPILED_GROUP9 = SUDOKU_DIR / "sudoku_group9_compiled.json"


def compile_4x4_exact(out: Path) -> Constraints:
    """Full 4x4 board compile under min-degree ordering, written to `out`.

    Same input and ordering as the committed sudoku_4x4_compiled.txt. Takes
    hours; each finished step is streamed to `out` as it completes, and the
    file is overwritten with the final artifact at the end.
    """
    cs = parse_constraints(generate_sudoku_dsl(4, value_base=1))
    cs.ordering = _heuristic_elimination_ordering(cs, "min-degree")
    print(f"compiling 4x4 board exactly -> {out} (expect hours)", flush=True)
    t0 = time.time()
    cc = compile_constraints(cs, stream_to=out)
    out.write_text(to_dsl(cc))
    n = sum(len(c) for c in cc.per_var_clauses)
    print(f"compiled {n} clauses in {time.time() - t0:.1f}s", flush=True)
    return Constraints.from_compiled(cc)


def build_layer(size: int, exact: bool, out: Path, rounds: int) -> nn.Module:
    """Compile or load the constraints for a `size` x `size` board and wrap them in a layer.

    4x4 is one exact CASPER layer over the whole board, loaded from the
    committed compile unless `exact` forces a recompile. 9x9 is KPassCasper:
    one compiled 9-clique queried for every cell against all 24 of its row,
    column and box peers, repeated for up to `rounds` rounds. A board counts as
    certified once a round makes no edits and hits no dead ends.
    """
    t0 = time.time()
    if size == 4:
        constraints = compile_4x4_exact(out) if exact else Constraints.from_compiled_dsl(COMPILED_4X4)
        layer = CasperLayer(constraints)
    else:
        layer = KPassCasper(Constraints.from_file(COMPILED_GROUP9), 9, max_rounds=rounds)
    print(f"layer ready in {time.time() - t0:.2f}s", flush=True)
    return layer


def valid_board(g: int, rng: torch.Generator) -> list[int]:
    """A valid g x g Sudoku (0-based digits, row-major) with randomly relabelled digits."""
    b = math.isqrt(g)
    perm = torch.randperm(g, generator=rng).tolist()
    return [perm[(b * (r % b) + r // b + c) % g] for r in range(g) for c in range(g)]


def synthetic_logits(
    g: int, n_corrupt: int, n_random: int, k: int, rng: torch.Generator
) -> tuple[torch.Tensor, list[str]]:
    """A batch of (B, g*g*g) logits standing in for a network's per-cell output.

    The first `n_corrupt` samples are a confident valid board with `k` cells
    flipped to a wrong digit (the true digit keeps some mass); the remaining
    `n_random` are unstructured random logits. Returns the logits and a
    per-sample kind label.
    """
    rows, kinds = [], []
    for _ in range(n_corrupt):
        logits = torch.zeros(g * g, g)
        board = valid_board(g, rng)
        logits[range(g * g), board] = 3.0
        for cell in torch.randperm(g * g, generator=rng)[:k].tolist():
            wrong = (board[cell] + 1 + torch.randint(g - 1, (1,), generator=rng).item()) % g
            logits[cell] = 0.0
            logits[cell, wrong] = 3.0
            logits[cell, board[cell]] = 1.5
        rows.append(logits.reshape(-1))
        kinds.append("corrupted")
    for _ in range(n_random):
        rows.append(torch.randn(g * g * g, generator=rng))
        kinds.append("random")
    return torch.stack(rows), kinds


def render(board: list[int], g: int) -> str:
    """Board pretty-printed with 1-based digits and box separators."""
    b = math.isqrt(g)
    lines = []
    for r in range(g):
        if r and r % b == 0:
            lines.append("-" * (2 * g + 2 * (b - 1) - 1))
        cells = []
        for c in range(g):
            if c and c % b == 0:
                cells.append("|")
            cells.append(str(board[r * g + c] + 1))
        lines.append(" ".join(cells))
    return "\n".join("    " + ln for ln in lines)


def report(pre: list[list[int]], post: list[list[int]], kinds: list[str], g: int) -> None:
    """Per-kind validity before/after, cells edited, and post-layer violations by group type."""
    groups = build_groups(g)
    for kind in dict.fromkeys(kinds):
        idx = [i for i, k in enumerate(kinds) if k == kind]
        v_pre = sum(not group_violations(pre[i], groups, g) for i in idx)
        v_post = sum(not group_violations(post[i], groups, g) for i in idx)
        edited = sum(a != b for i in idx for a, b in zip(pre[i], post[i]))
        bad = {"row": 0, "col": 0, "box": 0}
        for i in idx:
            for name in group_violations(post[i], groups, g):
                bad[name[:3]] += 1
        print(
            f"  {kind:>9}: valid {v_pre}/{len(idx)} -> {v_post}/{len(idx)}  "
            f"cells edited {edited}  violated groups after: "
            f"rows {bad['row']} cols {bad['col']} boxes {bad['box']}"
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compile/load a Sudoku CASPER layer and enforce it on a synthetic batch."
    )
    p.add_argument("--size", type=int, choices=(4, 9), default=4)
    p.add_argument("--exact", action="store_true",
                   help="4x4 only: recompile the full board (hours) instead of loading the committed compile.")
    p.add_argument("--out", type=Path, default=SUDOKU_DIR / "sudoku_4x4_exact_compiled.txt",
                   help="Where --exact writes its compile.")
    p.add_argument("--rounds", type=int, default=10, help="9x9 only: max KPassCasper rounds.")
    p.add_argument("--n-corrupt", type=int, default=6)
    p.add_argument("--n-random", type=int, default=2)
    p.add_argument("--flips", type=int, default=2, help="Wrong cells per corrupted board.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.exact and args.size != 4:
        p.error("--exact is only supported with --size 4; the full 9x9 compile is intractable")

    g = args.size
    layer = build_layer(g, args.exact, args.out, args.rounds)

    rng = torch.Generator().manual_seed(args.seed)
    logits, kinds = synthetic_logits(g, args.n_corrupt, args.n_random, args.flips, rng)
    logits.requires_grad_(True)
    B = logits.shape[0]

    probs = torch.softmax(logits.view(B, g * g, g), dim=-1).view(B, -1)
    t0 = time.time()
    adjusted, info = layer(probs)
    print(f"forward on B={B}: {time.time() - t0:.3f}s")

    pre = extract_assignments(probs, g * g, g)
    post = extract_assignments(adjusted, g * g, g)
    report(pre, post, kinds, g)
    if "certified_at" in info:
        print(f"  rounds used {info['rounds_used']}; certified at round: {info['certified_at']}")

    print("\n  sample 0 before:")
    print(render(pre[0], g))
    print("  sample 0 after:")
    print(render(post[0], g))

    info["projection_loss"].backward()
    print(
        f"\n  projection_loss={info['projection_loss'].item():.4f}  "
        f"|grad logits|={logits.grad.norm().item():.4f}"
    )


if __name__ == "__main__":
    main()
