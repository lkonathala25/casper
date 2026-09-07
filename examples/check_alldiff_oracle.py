from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "casper"))

from constraints import Constraints


def main() -> None:
    p = argparse.ArgumentParser(
        description="Artifact gate: the compiled feasible sets must equal the "
        "closed-form ALLDIFFERENT rule (domain minus already-used values)."
    )
    p.add_argument(
        "--artifact", type=Path,
        default=ROOT / "constraints" / "sudoku" / "sudoku_group9_compiled.json",
    )
    p.add_argument("--size", type=int, default=9)
    p.add_argument("--trials", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cons = Constraints.from_file(args.artifact)
    ordering = list(cons.ordering)
    n = len(ordering)
    domain = set(range(args.size))

    rng = random.Random(args.seed)
    mismatches: list[tuple] = []
    checked = 0
    empty_seen = 0

    for _ in range(args.trials):
        # A random legal prefix: distinct values assigned to a random-length
        # prefix of the ordering, which is exactly what INFER produces.
        k = rng.randint(0, n - 1)
        vals = rng.sample(sorted(domain), k)
        assignment = {ordering[i]: vals[i] for i in range(k)}

        got = cons.feasible_set(k, assignment)
        want = frozenset(domain - set(vals))
        checked += 1
        if not got:
            empty_seen += 1
        if got != want:
            mismatches.append((k, dict(assignment), sorted(got), sorted(want)))

    # Exhaustive sweep over every prefix length with a fixed canonical prefix,
    # so no step is left untested by sampling luck.
    for k in range(n):
        vals = list(range(k))
        assignment = {ordering[i]: vals[i] for i in range(k)}
        got = cons.feasible_set(k, assignment)
        want = frozenset(domain - set(vals))
        checked += 1
        if got != want:
            mismatches.append((k, dict(assignment), sorted(got), sorted(want)))

    print(f"artifact: {args.artifact}")
    print(f"checked {checked} prefixes; empty feasible sets: {empty_seen}")
    if mismatches:
        print(f"MISMATCHES: {len(mismatches)}")
        for k, a, g, w in mismatches[:5]:
            print(f"  step={k} prior={a}\n    got ={g}\n    want={w}")
    else:
        print("every compiled feasible set matched the closed-form rule")

    ok = not mismatches
    print(f"\nORACLE GATE: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
