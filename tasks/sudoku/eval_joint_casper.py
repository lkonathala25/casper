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
sys.path.insert(0, str(ROOT / "examples"))

from constraints import Constraints
from local_casper import (
    JointCasper,
    LocalGroupCasper,
    alldiff_group_constraints,
    extract_assignments,
)
from sudoku_groups import build_groups

from eval_local_casper_validity import (
    ARTIFACT,
    GROUPS,
    N,
    N_CELLS,
    build_batch,
    make_boards,
    violations,
)


def duplicate_offenders(board: list[int]) -> list[int]:
    """Cells that repeat a digit already placed earlier in one of their groups.

    Row-major visit order, so within each group the *first* holder of a
    repeated digit is blameless and every later one is an offender.
    """
    bad: set[int] = set()
    for _name, cells in GROUPS:
        seen: dict[int, int] = {}
        for cell in sorted(cells):
            v = board[cell]
            if v in seen:
                bad.add(cell)
            else:
                seen[v] = cell
    return sorted(bad)


def sweep27(layer, probs):
    """One 27-pass sweep; returns output and per-board edited-cell counts."""
    with torch.no_grad():
        out, info = layer(probs)
    edits = [0] * probs.shape[0]
    for e in info["edits_by_pass"]:
        for b, _c in e["cells"]:
            edits[b] += 1
    return out, edits


def main() -> None:
    p = argparse.ArgumentParser(
        description="Joint (row+col+box) conditioning sweep vs the 27-pass layer."
    )
    p.add_argument("--boards", type=int, default=20)
    p.add_argument("--per-k", type=int, default=100)
    p.add_argument("--max-sweeps", type=int, default=10)
    p.add_argument("--board-seed", type=int, default=20260731)
    p.add_argument("--corrupt-seed", type=int, default=12345)
    p.add_argument("--spot-check", type=int, default=10)
    p.add_argument("--out", type=Path, default=ROOT / "logs" / "joint_eval.json")
    args = p.parse_args()

    print(f"artifact  : {ARTIFACT}")
    print(f"seeds     : board={args.board_seed} corrupt={args.corrupt_seed}")
    print(f"joint mode: one row-major sweep, S = domain \\ (row U col U box)\n")

    seq = LocalGroupCasper(Constraints.from_file(ARTIFACT), GROUPS, N)
    joint_c = JointCasper(Constraints.from_file(ARTIFACT), N)
    joint_f = JointCasper(alldiff_group_constraints(N), N)

    bases = make_boards(args.boards, args.board_seed)
    HIGH = (0.80, 0.80, 0.15)
    rows = []

    for k in (1, 3, 5, 10, 15):
        # replicate the original harness's stream exactly
        rng = random.Random(args.corrupt_seed + 1000 * k)
        boards = [bases[rng.randrange(len(bases))] for _ in range(args.per_k)]
        probs, _corrupted = build_batch(boards, k, HIGH, rng)
        t0 = time.time()

        sl = probs[: args.spot_check]
        with torch.no_grad():
            oc, _ = joint_c(sl)
            of, _ = joint_f(sl)
        if not torch.equal(oc, of):
            raise SystemExit(f"HALT: compiled != from_function (joint) at k={k}")

        out1, _ = sweep27(seq, probs)
        post1 = extract_assignments(out1, N_CELLS, N)
        valid1 = [sum(violations(b).values()) == 0 for b in post1]

        cur = probs
        fixed = [None] * len(boards)
        for s in range(args.max_sweeps):
            cur, ed = sweep27(seq, cur)
            for i in range(len(boards)):
                if fixed[i] is None and ed[i] == 0:
                    fixed[i] = s + 1
        postI = extract_assignments(cur, N_CELLS, N)
        validI = [sum(violations(b).values()) == 0 for b in postI]

        with torch.no_grad():
            outJ, infoJ = joint_c(probs)
        postJ = extract_assignments(outJ, N_CELLS, N)
        validJ = [sum(violations(b).values()) == 0 for b in postJ]
        exactJ = [postJ[i] == boards[i] for i in range(len(boards))]
        de = infoJ["dead_ends"]

        # structural invariant: every offending cell must be a dead end
        for i, board in enumerate(postJ):
            offenders = duplicate_offenders(board)
            rogue = [c for c in offenders if c not in set(de[i])]
            if rogue:
                raise SystemExit(
                    f"HALT: non-dead-end violation at k={k} board {i}, "
                    f"cells {rogue[:5]} (dead ends were {de[i]})"
                )

        # self-certification: zero edits and zero dead ends implies validity
        pre = extract_assignments(probs, N_CELLS, N)
        for i in range(len(boards)):
            zero_edits = all(pre[i][c] == postJ[i][c] for c in range(N_CELLS))
            if zero_edits and not de[i] and not validJ[i]:
                raise SystemExit(
                    f"HALT: self-certification broken at k={k} board {i}"
                )

        affected = [len(d) for d in de if d]
        row = {
            "k": k,
            "valid_27_single": 100 * sum(valid1) / len(valid1),
            "valid_27_iter": 100 * sum(validI) / len(validI),
            "valid_joint": 100 * sum(validJ) / len(validJ),
            "exact_joint": 100 * sum(exactJ) / len(exactJ),
            "deadend_boards_pct": 100 * len(affected) / len(de),
            "deadend_mean_affected": (sum(affected) / len(affected)) if affected else 0.0,
            "resid_joint": sum(sum(violations(b).values()) for b in postJ) / len(postJ),
            "secs": round(time.time() - t0, 1),
        }
        rows.append(row)
        print(f"  k={k:>2}  27-pass {row['valid_27_single']:.0f}% / "
              f"{row['valid_27_iter']:.0f}%   joint {row['valid_joint']:.0f}%   "
              f"({row['secs']}s)", flush=True)

    hdr = (f"\n{'k':>3} {'27-single%':>11} {'27-iter%':>9} {'JOINT%':>8} "
           f"{'exactJ%':>8} {'DE boards%':>11} {'DE/affected':>12} {'residJ':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['k']:>3} {r['valid_27_single']:>11.0f} {r['valid_27_iter']:>9.0f} "
              f"{r['valid_joint']:>8.0f} {r['exact_joint']:>8.0f} "
              f"{r['deadend_boards_pct']:>11.0f} {r['deadend_mean_affected']:>12.2f} "
              f"{r['resid_joint']:>7.2f}")

    k1 = rows[0]
    print(f"\nPREDICTION CHECK (k=1, expected 85-90% joint):")
    print(f"  measured joint valid = {k1['valid_joint']:.0f}%  -> "
          f"{'PASS' if 85 <= k1['valid_joint'] <= 90 else 'OUTSIDE RANGE'}")

    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows,
                                    "board_seed": args.board_seed,
                                    "corrupt_seed": args.corrupt_seed}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
