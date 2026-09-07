from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
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
ARTIFACT = ROOT / "constraints" / "sudoku" / "sudoku_group9_compiled.json"
GROUPS = build_groups(N)
BOX_NAMES = {name for name, _ in GROUPS if name.startswith("box")}


def base_board() -> list[list[int]]:
    """The canonical shifted-rows Latin square, which is a valid Sudoku."""
    return [[(3 * (r % 3) + r // 3 + c) % 9 for c in range(N)] for r in range(N)]


def make_boards(n: int, seed: int) -> list[list[int]]:
    """`n` distinct valid boards via validity-preserving symmetries.

    Digit relabeling, row swaps inside a band, column swaps inside a stack,
    band and stack permutations, and transposition all map valid boards to
    valid boards, so every result is a genuine completed Sudoku.
    """
    rng = random.Random(seed)
    seen: set[tuple] = set()
    out: list[list[int]] = []
    while len(out) < n:
        g = [row[:] for row in base_board()]
        digits = list(range(N))
        rng.shuffle(digits)
        g = [[digits[v] for v in row] for row in g]
        for band in range(3):
            order = [0, 1, 2]
            rng.shuffle(order)
            rows = [g[band * 3 + o] for o in order]
            g[band * 3 : band * 3 + 3] = rows
        bands = [0, 1, 2]
        rng.shuffle(bands)
        g = [g[b * 3 + i] for b in bands for i in range(3)]
        g = [list(col) for col in zip(*g)]  # transpose -> column ops below
        for stack in range(3):
            order = [0, 1, 2]
            rng.shuffle(order)
            rows = [g[stack * 3 + o] for o in order]
            g[stack * 3 : stack * 3 + 3] = rows
        stacks = [0, 1, 2]
        rng.shuffle(stacks)
        g = [g[s * 3 + i] for s in stacks for i in range(3)]
        if rng.random() < 0.5:
            g = [list(col) for col in zip(*g)]
        flat = tuple(v for row in g for v in row)
        if flat in seen:
            continue
        seen.add(flat)
        out.append(list(flat))
    return out


def violations(board: list[int]) -> dict[str, int]:
    """Violated group counts split by row / col / box."""
    out = {"rows": 0, "cols": 0, "boxes": 0}
    key = {"r": "rows", "c": "cols", "b": "boxes"}
    for name, cells in GROUPS:
        if sorted(board[k] for k in cells) != list(range(N)):
            out[key[name[0]]] += 1
    return out


def build_batch(boards, k, profile, rng):
    """Score tensors for one (k, profile) cell of the sweep.

    Returns the probs tensor, the ground-truth boards, and the corrupted cell
    indices per board. Corrupted cells take a wrong digit, which on a valid
    board necessarily duplicates inside all three of that cell's groups.
    """
    peak, wrong_p, true_p = profile
    n = len(boards)
    other = (1.0 - peak) / (N - 1)
    probs = torch.full((n, N_CELLS, N), other, dtype=torch.float64)
    corrupted: list[list[int]] = []
    for b, board in enumerate(boards):
        for cell, val in enumerate(board):
            probs[b, cell, val] = peak
        cells = rng.sample(range(N_CELLS), k)
        corrupted.append(sorted(cells))
        rest = (1.0 - wrong_p - true_p) / (N - 2)
        for cell in cells:
            true = board[cell]
            wrong = rng.choice([d for d in range(N) if d != true])
            probs[b, cell, :] = rest
            probs[b, cell, wrong] = wrong_p
            probs[b, cell, true] = true_p
    return probs.reshape(n, N_CELLS * N), corrupted


def sweep(layer, probs):
    """One 27-pass sweep. Returns (output, per-board edited-cell counts)."""
    with torch.no_grad():
        out, info = layer(probs)
    n = probs.shape[0]
    edits = [0] * n
    for e in info["edits_by_pass"]:
        for b, _cell in e["cells"]:
            edits[b] += 1
    return out, edits


def main() -> None:
    p = argparse.ArgumentParser(
        description="Validity-rate characterization of the 27-pass local layer "
        "under synthetic corruptions."
    )
    p.add_argument("--boards", type=int, default=20)
    p.add_argument("--per-k", type=int, default=100)
    p.add_argument("--max-sweeps", type=int, default=10)
    p.add_argument("--board-seed", type=int, default=20260731)
    p.add_argument("--corrupt-seed", type=int, default=12345)
    p.add_argument("--spot-check", type=int, default=10)
    p.add_argument("--out", type=Path,
                   default=ROOT / "logs" / "validity_eval.json")
    args = p.parse_args()

    print(f"artifact       : {ARTIFACT}")
    print(f"board seed     : {args.board_seed}   corruption seed: {args.corrupt_seed}")
    print(f"base boards    : {args.boards}   boards per cell: {args.per_k}")
    print(f"pass order     : rows 1-9, cols 1-9, boxes 1-9 (27 passes)\n")

    compiled = LocalGroupCasper(Constraints.from_file(ARTIFACT), GROUPS, N)
    closed = LocalGroupCasper(alldiff_group_constraints(N), GROUPS, N)

    bases = make_boards(args.boards, args.board_seed)

    HIGH = (0.80, 0.80, 0.15)
    LOW = (0.80, 0.50, 0.35)
    configs = [(k, "high", HIGH) for k in (1, 3, 5, 10, 15)]
    configs += [(k, "low", LOW) for k in (3, 10)]

    rows = []
    curves = {}
    for k, pname, profile in configs:
        rng = random.Random(args.corrupt_seed + 1000 * k + (0 if pname == "high" else 7))
        boards = [bases[rng.randrange(len(bases))] for _ in range(args.per_k)]
        probs, corrupted = build_batch(boards, k, profile, rng)
        t0 = time.time()

        # spot-check tensor equality against the closed-form path
        sl = probs[: args.spot_check]
        with torch.no_grad():
            oc, _ = compiled(sl)
            of, _ = closed(sl)
        if not torch.equal(oc, of):
            raise SystemExit(
                f"HALT: compiled != from_function at k={k} profile={pname}"
            )

        out1, edits1 = sweep(compiled, probs)
        post1 = extract_assignments(out1, N_CELLS, N)
        pre = extract_assignments(probs, N_CELLS, N)

        v1 = [violations(b) for b in post1]
        if any(v["boxes"] for v in v1):
            bad = [i for i, v in enumerate(v1) if v["boxes"]]
            raise SystemExit(
                f"HALT: box violations after single sweep at k={k} "
                f"profile={pname}, boards {bad[:5]}"
            )

        valid1 = [sum(v.values()) == 0 for v in v1]
        exact1 = [post1[i] == boards[i] for i in range(len(boards))]
        moved1 = [
            sum(1 for c in range(N_CELLS) if pre[i][c] != post1[i][c])
            for i in range(len(boards))
        ]

        # iterated sweeps, recording the first zero-edit sweep per board
        cur = probs
        fixed_at = [None] * len(boards)
        curve = []
        for s in range(args.max_sweeps):
            cur, ed = sweep(compiled, cur)
            bo = extract_assignments(cur, N_CELLS, N)
            vs = [violations(b) for b in bo]
            curve.append(sum(sum(v.values()) for v in vs) / len(vs))
            for i in range(len(boards)):
                if fixed_at[i] is None and ed[i] == 0:
                    fixed_at[i] = s + 1
        postI = extract_assignments(cur, N_CELLS, N)
        vI = [violations(b) for b in postI]
        validI = [sum(v.values()) == 0 for v in vI]
        exactI = [postI[i] == boards[i] for i in range(len(boards))]

        # position-0 correlation: with rows first, a cell's first-touching
        # group is its row, and its position there is its column index
        def has_pos0(cells):
            return any(c % N == 0 for c in cells)

        base_rate = sum(has_pos0(c) for c in corrupted) / len(corrupted)
        fails = [i for i, ok in enumerate(valid1) if not ok]
        fail_rate = (
            sum(has_pos0(corrupted[i]) for i in fails) / len(fails) if fails else None
        )

        mv = {t: sum(v[t] for v in v1) / len(v1) for t in ("rows", "cols", "boxes")}
        row = {
            "k": k, "profile": pname,
            "valid_single": 100 * sum(valid1) / len(valid1),
            "valid_iterated": 100 * sum(validI) / len(validI),
            "exact_single": 100 * sum(exact1) / len(exact1),
            "exact_iterated": 100 * sum(exactI) / len(exactI),
            "resid_mean": sum(sum(v.values()) for v in v1) / len(v1),
            "resid_rows": mv["rows"], "resid_cols": mv["cols"],
            "resid_boxes": mv["boxes"],
            "moved_mean": sum(moved1) / len(moved1),
            "fixed_pct": 100 * sum(1 for f in fixed_at if f is not None) / len(fixed_at),
            "pos0_base_rate": 100 * base_rate,
            "pos0_among_failures": None if fail_rate is None else 100 * fail_rate,
            "n_failures": len(fails),
            "secs": round(time.time() - t0, 1),
        }
        rows.append(row)
        curves[f"k{k}_{pname}"] = curve
        print(f"  done k={k:>2} {pname:>4}  valid {row['valid_single']:.0f}% -> "
              f"{row['valid_iterated']:.0f}%   ({row['secs']}s)", flush=True)

    hdr = (f"\n{'k':>3} {'conf':>5} {'valid1%':>8} {'validIt%':>9} {'exact1%':>8} "
           f"{'exactIt%':>9} {'resid':>7} {'r/c/b':>14} {'moved':>7} {'fixpt%':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        rcb = f"{r['resid_rows']:.1f}/{r['resid_cols']:.1f}/{r['resid_boxes']:.1f}"
        print(f"{r['k']:>3} {r['profile']:>5} {r['valid_single']:>8.0f} "
              f"{r['valid_iterated']:>9.0f} {r['exact_single']:>8.0f} "
              f"{r['exact_iterated']:>9.0f} {r['resid_mean']:>7.2f} {rcb:>14} "
              f"{r['moved_mean']:>7.2f} {r['fixed_pct']:>7.0f}")

    print("\nviolations-per-sweep (mean per board, sweeps 1..N):")
    for key, c in curves.items():
        print(f"  {key:>10}: " + " ".join(f"{x:5.2f}" for x in c))

    print("\nposition-0 correlation (corrupted cell first in its row pass):")
    print(f"{'k':>3} {'conf':>5} {'base rate%':>11} {'among failures%':>17} {'n_fail':>7}")
    for r in rows:
        af = "n/a" if r["pos0_among_failures"] is None else f"{r['pos0_among_failures']:.0f}"
        print(f"{r['k']:>3} {r['profile']:>5} {r['pos0_base_rate']:>11.0f} "
              f"{af:>17} {r['n_failures']:>7}")

    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(
        {"rows": rows, "curves": curves,
         "board_seed": args.board_seed, "corrupt_seed": args.corrupt_seed,
         "boards": bases}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
