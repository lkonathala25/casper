from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _sub in ("casper", "tasks/sudoku"):
    if str(ROOT / _sub) not in sys.path:
        sys.path.insert(0, str(ROOT / _sub))

from resolution import (
    _dedup_by_subsumption,
    _enumerate_minimal_conflict_patterns,
    _grouped_eliminate_var,
    _is_tautology,
)
from signed_clauses import parse_constraints
from sudoku_groups import group_dsl


def advance_to_step(cs, target_step: int, verbose: bool = False):
    """Run the elimination loop down to `target_step` and return its pi_current.

    Mirrors the body of `compile_constraints` so the state handed to the
    analysis is the same state the compiler would see at that step.
    """
    domains = list(cs.var_domains)
    ordering = list(cs.ordering)
    n = len(ordering)
    pi_current = [
        c for c in cs.clauses if c.literals and not _is_tautology(c, domains)
    ]

    for step in range(n - 1, target_step, -1):
        var = ordering[step]
        pi_bar = [c for c in pi_current if var in c.variables]
        pi_minus_bar = [c for c in pi_current if var not in c.variables]
        if not pi_bar:
            continue
        t0 = time.time()
        kept, _stats = _grouped_eliminate_var(
            pi_bar, pi_minus_bar, var, domains, cap=domains[var]
        )
        seen = set(pi_minus_bar)
        next_pi = list(pi_minus_bar)
        for c in kept:
            if c not in seen:
                seen.add(c)
                next_pi.append(c)
        pi_current = _dedup_by_subsumption(next_pi)
        if verbose:
            print(
                f"  eliminated step={step} var={cs.var_names[var]} "
                f"-> pi_current={len(pi_current)} ({time.time() - t0:.1f}s)",
                flush=True,
            )
    return pi_current, domains, ordering


def canonical_form(sets: list[frozenset[int]], n_values: int) -> tuple:
    """Canonical form of a family of value-sets under relabeling of values.

    Fixing an order on the k sets gives every value a k-bit membership mask;
    the family is then determined up to value relabeling by the multiset of
    those masks. Minimising the sorted mask multiset over all k! set orders
    quotients out the set ordering too, so two families share a canonical form
    exactly when some value relabeling carries one onto the other.
    """
    k = len(sets)
    best: tuple | None = None
    for order in permutations(range(k)):
        masks = []
        for v in range(n_values):
            m = 0
            for bit, j in enumerate(order):
                if v in sets[j]:
                    m |= 1 << bit
            masks.append(m)
        cand = tuple(sorted(masks))
        if best is None or cand < best:
            best = cand
    return best


def main() -> None:
    p = argparse.ArgumentParser(
        description="Canonicalise the minimal conflict patterns at one "
        "elimination step under value relabeling and count the orbits."
    )
    p.add_argument("--size", type=int, default=9)
    p.add_argument("--step", type=int, default=6)
    p.add_argument("--cap", type=int, default=4)
    args = p.parse_args()

    cs = parse_constraints(group_dsl(args.size))
    print(f"advancing to step={args.step} ...", flush=True)
    pi_current, domains, ordering = advance_to_step(cs, args.step, verbose=True)

    var = ordering[args.step]
    var_name = cs.var_names[var]
    pi_bar = [c for c in pi_current if var in c.variables]

    groups: dict[frozenset[int], list[int]] = defaultdict(list)
    for j, c in enumerate(pi_bar):
        groups[c.get(var) or frozenset()].append(j)
    distinct_sets = list(groups.keys())

    print(
        f"\nstep={args.step} var={var_name} pi_bar={len(pi_bar)} "
        f"distinct_value_sets={len(distinct_sets)} cap={args.cap}",
        flush=True,
    )

    t0 = time.time()
    patterns = _enumerate_minimal_conflict_patterns(distinct_sets, args.cap)
    print(
        f"enumerated {len(patterns)} minimal patterns "
        f"(sizes 1..{args.cap}) in {time.time() - t0:.1f}s\n",
        flush=True,
    )

    by_size: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for J in patterns:
        by_size[len(J)].append(J)

    n_values = args.size
    grand: Counter = Counter()
    print(f"{'size':>5} {'patterns':>10} {'canonical forms':>16} {'ratio':>8}")
    for k in sorted(by_size):
        forms = Counter(
            canonical_form([distinct_sets[j] for j in J], n_values)
            for J in by_size[k]
        )
        grand.update(forms)
        n = len(by_size[k])
        print(f"{k:>5} {n:>10} {len(forms):>16} {n / len(forms):>8.1f}x")

    total = sum(grand.values())
    print(
        f"\nTOTAL: {total} patterns -> {len(grand)} distinct canonical forms "
        f"({total / len(grand):.1f}x compression)"
    )

    print("\norbit sizes (how many patterns share each canonical form):")
    for form, cnt in grand.most_common(15):
        popcounts = sorted(bin(m).count("1") for m in form)
        print(f"  orbit={cnt:>7}  value-degree profile={popcounts}")
    if len(grand) > 15:
        print(f"  ... {len(grand) - 15} more forms")


if __name__ == "__main__":
    main()
