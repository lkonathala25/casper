from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _sub in ("casper", "tasks/sudoku"):
    if str(ROOT / _sub) not in sys.path:
        sys.path.insert(0, str(ROOT / _sub))

from resolution import (
    _dedup_by_subsumption,
    _disjoin,
    _enumerate_minimal_conflict_patterns_orbit,
    _grouped_eliminate_var,
    _is_tautology,
    value_generators,
    permute_clause_values,
)
from signed_clauses import parse_constraints
from sudoku_groups import group_dsl


def expand_pattern(
    J: tuple[int, ...],
    distinct_sets: list[frozenset[int]],
    groups: dict[frozenset[int], list[int]],
    pi_bar: list,
    var: int,
    domains: list[int],
) -> set:
    """Every resolvent of one pattern, straight from the definition.

    Cross-product over the pattern's groups, disjoining one dropped clause from
    each, discarding tautologies. Deliberately independent of the optimised
    expander so the gate cannot inherit a bug from the code it validates.
    """
    choices = [groups[distinct_sets[i]] for i in J]
    out = set()
    for combo in product(*choices):
        disj = _disjoin([pi_bar[c].drop(var) for c in combo])
        if disj.literals and not _is_tautology(disj, domains):
            out.add(disj)
    return out


def check_step(pi, var, domains, n_values, label, sample, max_expansion):
    """Verify expand(g.J) == g.expand(J) on real patterns from this step."""
    pi_bar = [c for c in pi if var in c.variables]
    groups: dict[frozenset[int], list[int]] = defaultdict(list)
    for j, c in enumerate(pi_bar):
        groups[c.get(var) or frozenset()].append(j)
    distinct = list(groups.keys())
    idx_of = {s: i for i, s in enumerate(distinct)}

    patterns = _enumerate_minimal_conflict_patterns_orbit(
        distinct, domains[var], n_values
    )

    checked = 0
    skipped = 0
    failures = []
    for g in value_generators(n_values):
        gmap = [idx_of[frozenset(g[v] for v in s)] for s in distinct]
        step = max(1, len(patterns) // sample)
        for J in patterns[::step]:
            size = 1
            for i in J:
                size *= len(groups[distinct[i]])
            if size > max_expansion:
                skipped += 1
                continue
            gJ = tuple(sorted(gmap[i] for i in J))
            lhs = expand_pattern(gJ, distinct, groups, pi_bar, var, domains)
            rhs = {
                permute_clause_values(c, g)
                for c in expand_pattern(J, distinct, groups, pi_bar, var, domains)
            }
            checked += 1
            if lhs != rhs:
                failures.append((J, gJ, len(lhs), len(rhs)))

    status = "PASS" if not failures else "FAIL"
    print(
        f"{label}: patterns={len(patterns)} checked={checked} "
        f"skipped_oversize={skipped} mismatches={len(failures)} -> {status}",
        flush=True,
    )
    for J, gJ, a, b in failures[:3]:
        print(f"    J={J} gJ={gJ} |expand(gJ)|={a} |g.expand(J)|={b}")
    return not failures


def main() -> None:
    p = argparse.ArgumentParser(
        description="Shipping gate: expand(g.J) must equal g.expand(J) on real "
        "patterns from completed elimination steps."
    )
    p.add_argument("--size", type=int, default=9)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--sample", type=int, default=150)
    p.add_argument("--max-expansion", type=int, default=20000)
    args = p.parse_args()

    cs = parse_constraints(group_dsl(args.size))
    domains = list(cs.var_domains)
    ordering = list(cs.ordering)
    n = len(ordering)

    pi = [c for c in cs.clauses if c.literals and not _is_tautology(c, domains)]
    ok = True
    for step in range(n - 1, n - 1 - args.steps, -1):
        var = ordering[step]
        ok &= check_step(
            pi, var, domains, args.size,
            f"step={step} var={cs.var_names[var]}",
            args.sample, args.max_expansion,
        )
        pm = [c for c in pi if var not in c.variables]
        pb = [c for c in pi if var in c.variables]
        kept, _ = _grouped_eliminate_var(pb, pm, var, domains, cap=domains[var])
        seen = set(pm)
        nxt = list(pm)
        for c in kept:
            if c not in seen:
                seen.add(c)
                nxt.append(c)
        pi = _dedup_by_subsumption(nxt)

    print(f"\nRESOLVENT IDENTITY GATE: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
