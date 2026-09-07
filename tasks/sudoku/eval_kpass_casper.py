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
    KPassCasper,
    alldiff_group_constraints,
    extract_assignments,
)

from eval_local_casper_validity import (
    ARTIFACT,
    N,
    N_CELLS,
    build_batch,
    make_boards,
    violations,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="K-round all-peers ('memory') mode vs the joint 1-sweep mode."
    )
    p.add_argument("--boards", type=int, default=20)
    p.add_argument("--per-k", type=int, default=100)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--board-seed", type=int, default=20260731)
    p.add_argument("--corrupt-seed", type=int, default=12345)
    p.add_argument("--spot-check", type=int, default=10)
    p.add_argument("--out", type=Path, default=ROOT / "logs" / "kpass_eval.json")
    args = p.parse_args()

    print(f"artifact : {ARTIFACT}")
    print(f"seeds    : board={args.board_seed} corrupt={args.corrupt_seed}")
    print(f"K-pass   : <= {args.rounds} rounds over 27 blocks, all-peers "
          f"conditioning, dead ends retried\n")

    joint = JointCasper(Constraints.from_file(ARTIFACT), N)
    kp_c = KPassCasper(Constraints.from_file(ARTIFACT), N, args.rounds)
    kp_f = KPassCasper(alldiff_group_constraints(N), N, args.rounds)

    bases = make_boards(args.boards, args.board_seed)
    HIGH = (0.80, 0.80, 0.15)
    rows = []

    for k in (1, 3, 5, 10, 15):
        rng = random.Random(args.corrupt_seed + 1000 * k)
        boards = [bases[rng.randrange(len(bases))] for _ in range(args.per_k)]
        probs, _ = build_batch(boards, k, HIGH, rng)
        t0 = time.time()

        sl = probs[: args.spot_check]
        with torch.no_grad():
            a, _ = kp_c(sl)
            b, _ = kp_f(sl)
        if not torch.equal(a, b):
            raise SystemExit(f"HALT: compiled != from_function (K-pass) at k={k}")

        with torch.no_grad():
            outJ, _ = joint(probs)
        postJ = extract_assignments(outJ, N_CELLS, N)
        validJ = [sum(violations(x).values()) == 0 for x in postJ]

        with torch.no_grad():
            outK, infoK = kp_c(probs)
        postK = extract_assignments(outK, N_CELLS, N)
        validK = [sum(violations(x).values()) == 0 for x in postK]
        exactK = [postK[i] == boards[i] for i in range(len(boards))]
        cert = infoK["certified_at"]
        weak = infoK["weak_certified_at"]

        # zero tolerance: anything that certifies must be fully valid
        for i, c in enumerate(cert):
            if c is not None and not validK[i]:
                raise SystemExit(
                    f"HALT: board {i} certified at round {c} but is invalid (k={k})"
                )
        # how often would the edits-only rule have been wrong?
        weak_wrong = sum(
            1 for i, w in enumerate(weak)
            if w is not None and cert[i] is None and not validK[i]
        )

        certified = [c for c in cert if c is not None]
        final_dead = infoK["dead_ends_per_round"][-1]
        plateau = [
            i for i in range(len(boards))
            if cert[i] is None and infoK["edits_per_round"][-1][i] > 0
        ]
        curve = [
            sum(r) / len(r) for r in infoK["edits_per_round"]
        ]

        row = {
            "k": k,
            "valid_joint": 100 * sum(validJ) / len(validJ),
            "valid_kpass": 100 * sum(validK) / len(validK),
            "exact_kpass": 100 * sum(exactK) / len(exactK),
            "cert_pct": 100 * len(certified) / len(cert),
            "mean_rounds_to_cert": (sum(certified) / len(certified)) if certified else None,
            "dead_final_mean": sum(final_dead) / len(final_dead),
            "plateau_pct": 100 * len(plateau) / len(boards),
            "weak_rule_wrong": weak_wrong,
            "rounds_used": infoK["rounds_used"],
            "edits_curve": [round(x, 2) for x in curve],
            "secs": round(time.time() - t0, 1),
        }
        rows.append(row)
        mr = "n/a" if row["mean_rounds_to_cert"] is None else f"{row['mean_rounds_to_cert']:.2f}"
        print(f"  k={k:>2}  joint {row['valid_joint']:.0f}%  ->  K-pass "
              f"{row['valid_kpass']:.0f}%   cert {row['cert_pct']:.0f}% "
              f"@ mean {mr} rounds   ({row['secs']}s)", flush=True)

    hdr = (f"\n{'k':>3} {'joint%':>7} {'KPASS%':>8} {'exactK%':>8} {'cert%':>7} "
           f"{'mean rounds':>12} {'deadEnd fin':>12} {'plateau%':>9} {'weakBad':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        mr = "n/a" if r["mean_rounds_to_cert"] is None else f"{r['mean_rounds_to_cert']:.2f}"
        print(f"{r['k']:>3} {r['valid_joint']:>7.0f} {r['valid_kpass']:>8.0f} "
              f"{r['exact_kpass']:>8.0f} {r['cert_pct']:>7.0f} {mr:>12} "
              f"{r['dead_final_mean']:>12.2f} {r['plateau_pct']:>9.0f} "
              f"{r['weak_rule_wrong']:>8}")

    print("\nedits per round (mean per board):")
    for r in rows:
        print(f"  k={r['k']:>2}: " + " ".join(f"{x:6.2f}" for x in r["edits_curve"]))

    print("\nPRE-REGISTERED PREDICTIONS")
    g = {r["k"]: r["valid_kpass"] - r["valid_joint"] for r in rows}
    p1 = all(g[k] >= g[1] for k in (3, 5, 10, 15))
    print(f"  (1) gain concentrated at k>=3: deltas "
          f"{ {k: round(v) for k, v in g.items()} }  -> {'PASS' if p1 else 'FAIL'}")
    osc = [r["k"] for r in rows if r["plateau_pct"] > 0]
    print(f"  (2) oscillation returns with revisiting: plateaued boards at k="
          f"{osc if osc else 'none'}  -> {'observed' if osc else 'not observed'}")

    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows,
                                    "board_seed": args.board_seed,
                                    "corrupt_seed": args.corrupt_seed}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
