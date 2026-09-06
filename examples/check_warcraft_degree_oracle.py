from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from constraints import Constraints  # noqa: E402
from warcraft_degree_constraints import node_incident  # noqa: E402


def brute_force_models(N: int) -> set[tuple[int, ...]]:
    """Every edge subset with endpoint degree 1 and all other degrees in {0,2}."""
    inc = node_incident(N)
    n_edges = 2 * N * (N - 1)
    nodes = sorted(inc)
    A = np.zeros((len(nodes), n_edges), dtype=np.int8)
    for r, node in enumerate(nodes):
        for e in inc[node]:
            A[r, e] = 1
    endpoint = np.array([node in ((0, 0), (N - 1, N - 1)) for node in nodes])

    total = 1 << n_edges
    chunk = min(total, 1 << 20)
    bits = np.arange(n_edges, dtype=np.int64)
    out: set[tuple[int, ...]] = set()
    for start in range(0, total, chunk):
        idx = np.arange(start, min(start + chunk, total), dtype=np.int64)
        X = ((idx[:, None] >> bits[None, :]) & 1).astype(np.int8)
        deg = X @ A.T
        ok = np.ones(len(idx), dtype=bool)
        ok &= (deg[:, endpoint] == 1).all(axis=1)
        other = deg[:, ~endpoint]
        ok &= ((other == 0) | (other == 2)).all(axis=1)
        for row in X[ok]:
            out.add(tuple(int(v) for v in row))
    return out


def walk_models(cons: Constraints, n_edges: int):
    """Every assignment the compiled artifact admits, plus a dead-end count.

    A dead end is a value the compiled feasible set admits at some prefix that
    extends to no complete model — the compile would be incomplete.
    """
    ordering = list(cons.ordering)
    n = len(ordering)
    models: set[tuple[int, ...]] = set()
    dead_ends: list[tuple[int, dict, int]] = []

    def rec(step: int, assignment: dict[int, int]) -> int:
        if step == n:
            models.add(tuple(assignment[v] for v in range(n_edges)))
            return 1
        found = 0
        for val in sorted(cons.feasible_set(step, assignment)):
            assignment[ordering[step]] = val
            got = rec(step + 1, assignment)
            if got == 0:
                dead_ends.append((step, dict(assignment), val))
            found += got
            del assignment[ordering[step]]
        return found

    rec(0, {})
    return models, dead_ends


def main() -> None:
    p = argparse.ArgumentParser(
        description="Oracle gate: compiled feasible sets must admit exactly the "
        "degree-legal edge sets found by brute force."
    )
    p.add_argument("--grid", type=int, required=True)
    p.add_argument("--artifact", type=Path, required=True)
    args = p.parse_args()

    N = args.grid
    n_edges = 2 * N * (N - 1)
    cons = Constraints.from_file(args.artifact)

    print(f"grid {N}x{N}  edges={n_edges}  artifact={args.artifact.name}")

    want = brute_force_models(N)
    got, dead_ends = walk_models(cons, n_edges)

    missing = want - got
    extra = got - want

    print(f"  brute-force degree-legal edge sets : {len(want)}")
    print(f"  admitted by compiled artifact      : {len(got)}")
    print(f"  missing (valid but rejected)       : {len(missing)}")
    print(f"  extra   (invalid but admitted)     : {len(extra)}")
    print(f"  dead-end feasible values           : {len(dead_ends)}")

    for label, s in (("missing", missing), ("extra", extra)):
        for m in list(s)[:3]:
            on = [i for i, v in enumerate(m) if v]
            print(f"    {label} example: on-edges {on}")

    ok = not missing and not extra and not dead_ends
    print(f"  ORACLE GATE: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
