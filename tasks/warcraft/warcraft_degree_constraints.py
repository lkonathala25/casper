from __future__ import annotations

import argparse
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "casper"))


def edge_index_maps(N: int):
    """Canonical edge indexing: horizontal edges row-major, then vertical.

    Matches `edges.npy` from warcraft_preprocess.py and the right_idx/down_idx
    helpers in src_warcraft/node_categories.py.
    """
    n_w = N - 1
    n_right = N * n_w

    def right_idx(i, j):
        return i * n_w + j

    def down_idx(i, j):
        return n_right + i * N + j

    edges = [None] * (2 * N * (N - 1))
    for i in range(N):
        for j in range(N - 1):
            edges[right_idx(i, j)] = (i, j, i, j + 1)
    for i in range(N - 1):
        for j in range(N):
            edges[down_idx(i, j)] = (i, j, i + 1, j)
    return edges, right_idx, down_idx


def node_incident(N: int) -> dict[tuple[int, int], list[int]]:
    """Node -> sorted list of incident edge indices."""
    _, right_idx, down_idx = edge_index_maps(N)
    inc: dict[tuple[int, int], list[int]] = {}
    for i in range(N):
        for j in range(N):
            e = []
            if j > 0:
                e.append(right_idx(i, j - 1))
            if j < N - 1:
                e.append(right_idx(i, j))
            if i > 0:
                e.append(down_idx(i - 1, j))
            if i < N - 1:
                e.append(down_idx(i, j))
            inc[(i, j)] = sorted(e)
    return inc


def degree_clauses(N: int):
    """Family 1 degree constraints as (clause, family) pairs.

    Endpoints carry exactly one incident edge; every other node carries zero or
    two. Each clause is a list of (edge_index, value) literals, <= 4 vars wide.
    """
    inc = node_incident(N)
    source, sink = (0, 0), (N - 1, N - 1)
    out = []
    for node, edges in inc.items():
        d = len(edges)
        if node in (source, sink):
            out.append(([(e, 1) for e in edges], f"exactly_one_deg{d}"))
            for a, b in combinations(edges, 2):
                out.append(([(a, 0), (b, 0)], f"exactly_one_deg{d}"))
        else:
            # Forbid degree 1: if e is on, some other incident edge is on.
            for e in edges:
                lits = [(e, 0)] + [(f, 1) for f in edges if f != e]
                out.append((lits, f"zero_or_two_deg{d}"))
            # Forbid degree >= 3: no three incident edges all on.
            for tri in combinations(edges, 3):
                out.append(([(e, 0) for e in tri], f"zero_or_two_deg{d}"))
    return out


def build_ordering(N: int, mode: str) -> list[int]:
    """INFER assignment order over edge variables; elimination is its reverse."""
    n_w = N - 1
    n_right = N * n_w
    if mode == "index":
        return list(range(2 * N * (N - 1)))
    if mode == "geometric":
        # Row by row: that row's horizontal edges, then the verticals leaving it.
        order = []
        for i in range(N):
            order.extend(i * n_w + j for j in range(N - 1))
            if i < N - 1:
                order.extend(n_right + i * N + j for j in range(N))
        return order
    raise ValueError(mode)


def to_dsl_text(N: int, ordering: list[int]) -> tuple[str, Counter]:
    n_edges = 2 * N * (N - 1)
    clauses = degree_clauses(N)
    fams = Counter(f for _, f in clauses)

    lines = [f"var y_{e} 2" for e in range(n_edges)]
    lines.append("")
    lines.append("ordering " + " ".join(f"y_{e}" for e in ordering))
    lines.append("")
    for lits, _ in clauses:
        lines.append(" or ".join(f"[{v}]:y_{e}" for e, v in lits))
    return "\n".join(lines) + "\n", fams


def induced_width(N: int, ordering: list[int]) -> tuple[int, list[int]]:
    """Induced width of the primal graph under elimination = reverse(ordering).

    The primal graph is the line graph of the grid: two edge-variables are
    adjacent iff they share a node. Returns (width, per-step frontier sizes).
    """
    inc = node_incident(N)
    adj: dict[int, set[int]] = {e: set() for e in range(2 * N * (N - 1))}
    for edges in inc.values():
        for a, b in combinations(edges, 2):
            adj[a].add(b)
            adj[b].add(a)

    alive = set(adj)
    widths = []
    for v in reversed(ordering):
        nbrs = adj[v] & alive
        widths.append(len(nbrs))
        for a, b in combinations(sorted(nbrs), 2):
            adj[a].add(b)
            adj[b].add(a)
        alive.discard(v)
    return (max(widths) if widths else 0), widths


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate Family 1 (degree) constraints for the Warcraft grid."
    )
    p.add_argument("--grid", type=int, required=True)
    p.add_argument("--ordering", choices=("index", "geometric"), default="index")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--width-only", action="store_true")
    args = p.parse_args()

    N = args.grid
    n_edges = 2 * N * (N - 1)

    if args.width_only:
        print(f"grid {N}x{N}  edges={n_edges}")
        for mode in ("index", "geometric"):
            w, _ = induced_width(N, build_ordering(N, mode))
            print(f"  ordering={mode:<10} induced_width={w:<4} "
                  f"(worst-case clauses ~ 2^{w})")
        try:
            from resolution import _heuristic_elimination_ordering
            from signed_clauses import parse_constraints
            text, _ = to_dsl_text(N, build_ordering(N, "index"))
            cs = parse_constraints(text)
            for h in ("min-degree", "min-fill"):
                order = _heuristic_elimination_ordering(cs, h)
                w, _ = induced_width(N, order)
                print(f"  ordering={h:<10} induced_width={w:<4} "
                      f"(worst-case clauses ~ 2^{w})")
        except Exception as e:
            print(f"  (heuristic orderings unavailable: {e})")
        return

    text, fams = to_dsl_text(N, build_ordering(N, args.ordering))
    n_clauses = sum(fams.values())
    print(f"grid {N}x{N}  variables={n_edges}  clauses={n_clauses}")
    for k, v in sorted(fams.items()):
        print(f"  {k}: {v}")
    w, _ = induced_width(N, build_ordering(N, args.ordering))
    print(f"  ordering={args.ordering}  induced_width={w}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
