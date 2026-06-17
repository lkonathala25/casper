#!/usr/bin/env python3
"""Scope the cross-variable parallelism available in a signed-resolution compile.

Bucket elimination feeds each variable's bucket from (a) original clauses on it
and (b) resolvents deposited by variables eliminated earlier. The dependency
structure is the *elimination tree* (etree) of the min-degree-filled primal
graph: a variable's resolvent flows to its etree parent, so a variable receives
"late" clauses only from its etree children.

For each variable v this reports:
  K  = original clauses mentioning v        (known up front -> speculatable now)
  L? = does v have etree children           (resolvents that must arrive later)
and the tree-level metrics that bound cross-variable parallelism:
  critical path (etree height) = sequential lower bound even with infinite cores
  leaves                       = the initial fully-independent elimination front
  antichain width per level    = how many variables can be in flight together

Usage:
    python3 examples/analyze_etree.py examples/sudoku_9x9.txt
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from signed_clauses import parse_constraints  # noqa: E402
from resolution import _primal_adjacency  # noqa: E402


def min_degree_fill(adj, n):
    """Simulate min-degree elimination, returning the elimination sequence
    (first-eliminated first) and each variable's bucket variables (its filled
    neighbours still remaining at elimination time = the resolvent's support)."""
    adj = [set(ns) for ns in adj]
    remaining = set(range(n))
    elim_seq = []
    bucket_vars = {}
    while remaining:
        v = min(remaining, key=lambda x: len(adj[x] & remaining))
        nb = adj[v] & remaining
        bucket_vars[v] = set(nb)
        for a in nb:
            adj[a] |= (nb - {a})
        remaining.discard(v)
        elim_seq.append(v)
    return elim_seq, bucket_vars


def main():
    path = sys.argv[1]
    cs = parse_constraints(Path(path).read_text())
    n = len(cs.var_domains)
    names = cs.var_names

    adj = _primal_adjacency(cs)
    elim_seq, bucket_vars = min_degree_fill(adj, n)
    pos = {v: i for i, v in enumerate(elim_seq)}  # 0 = eliminated first

    # etree parent(v) = earliest-eliminated among v's later (filled) neighbours.
    parent = {}
    children = defaultdict(list)
    for v in elim_seq:
        later = bucket_vars[v]  # all eliminated after v by construction
        if later:
            p = min(later, key=lambda u: pos[u])
            parent[v] = p
            children[p].append(v)
        else:
            parent[v] = None

    # Original clause count per variable (= K, the speculatable bucket portion).
    orig_clauses = defaultdict(int)
    for c in cs.clauses:
        for var in c.variables:
            orig_clauses[var] += 1

    # Depth from leaves: leaf level 0; a node is one above its deepest child.
    level = {}
    for v in elim_seq:  # children precede parents in elim_seq
        level[v] = 0 if not children[v] else 1 + max(level[u] for u in children[v])
    height = max(level.values()) + 1
    width_per_level = defaultdict(int)
    for v in elim_seq:
        width_per_level[level[v]] += 1
    leaves = sum(1 for v in elim_seq if not children[v])

    print(f"\n=== etree scope: {path}  (n_vars={n}) ===")
    print(f"critical path (etree height) : {height}    "
          f"# sequential steps with infinite cores")
    print(f"initial independent leaves   : {leaves}    "
          f"# fully-K variables, eliminable concurrently")
    print(f"max level width              : {max(width_per_level.values())}")
    print(f"mean children (internal)     : "
          f"{sum(len(children[v]) for v in elim_seq) / max(1, sum(1 for v in elim_seq if children[v])):.2f}")

    # The bottleneck question: is the 2nd-eliminated var independent of the 1st?
    first, second = elim_seq[0], elim_seq[1]
    feeds = parent[first] == second
    print(f"\nfirst eliminated  : {names[first]}  "
          f"(K={orig_clauses[first]} clauses, children={len(children[first])})")
    print(f"second eliminated : {names[second]} "
          f"(K={orig_clauses[second]} clauses, children={len(children[second])})")
    print(f"  -> {names[first]} resolvent feeds {names[second]}'s bucket? "
          f"{'YES (dependent)' if feeds else 'NO (independent)'}")

    # First 15 eliminated: how much is speculatable (K) vs needs children (L).
    print(f"\n{'elim#':>5} {'var':>6} {'K(orig)':>8} {'children(L src)':>15} "
          f"{'level':>6}")
    for i, v in enumerate(elim_seq[:15]):
        kids = children[v]
        ksrc = ",".join(names[u] for u in sorted(kids, key=lambda u: pos[u])[:4])
        ksrc = f"{len(kids)} [{ksrc}{'...' if len(kids) > 4 else ''}]" if kids else "0 (pure K)"
        print(f"{i:>5} {names[v]:>6} {orig_clauses[v]:>8} {ksrc:>15} {level[v]:>6}")

    # How many variables are "pure K" (etree leaves) = free parallelism front.
    pure_k = [v for v in elim_seq if not children[v]]
    print(f"\npure-K variables (no late deps, eliminable immediately): "
          f"{len(pure_k)} / {n}")
    print(f"  fraction of compile that is embarrassingly parallel up front: "
          f"{len(pure_k) / n:.0%}")


if __name__ == "__main__":
    main()
