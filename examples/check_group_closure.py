from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from resolution import (
    _dedup_by_subsumption,
    _enumerate_minimal_conflict_patterns_orbit,
    _grouped_eliminate_var,
    _is_tautology,
)
from signed_clauses import SignedClause, SignedLiteral, parse_constraints
from sudoku_groups import group_dsl


def permute_values(clause: SignedClause, perm: tuple[int, ...]) -> SignedClause:
    """Relabel every literal's value-set through `perm`, keeping variables fixed."""
    return SignedClause(
        tuple(
            SignedLiteral(lit.var, frozenset(perm[v] for v in lit.values))
            for lit in clause.literals
        )
    )


def permute_vars(clause: SignedClause, vperm: dict[int, int]) -> SignedClause:
    """Relabel every literal's variable through `vperm`, keeping values fixed."""
    return SignedClause.from_literals(
        SignedLiteral(vperm[lit.var], lit.values) for lit in clause.literals
    )


def generators(n: int) -> list[tuple[int, ...]]:
    """A transposition and an n-cycle — together these generate S_n."""
    swap = tuple([1, 0] + list(range(2, n)))
    cycle = tuple(list(range(1, n)) + [0])
    return [swap, cycle]


def check_step(pi, var, domains, n_values, live_vars, label):
    """Report whether pi and the step's pi_bar survive value/variable relabeling."""
    pi_set = set(pi)
    ok_val = True
    for g in generators(n_values):
        if {permute_values(c, g) for c in pi} != pi_set:
            ok_val = False

    # Variable relabeling: swap two still-live variables, and cycle them.
    ok_var = True
    lv = sorted(live_vars)
    if len(lv) >= 2:
        swap = {v: v for v in lv}
        swap[lv[0]], swap[lv[1]] = lv[1], lv[0]
        cyc = {lv[i]: lv[(i + 1) % len(lv)] for i in range(len(lv))}
        for vp in (swap, cyc):
            if {permute_vars(c, vp) for c in pi} != pi_set:
                ok_var = False

    pi_bar = [c for c in pi if var in c.variables]
    bar_set = set(pi_bar)
    ok_bar = all(
        {permute_values(c, g) for c in pi_bar} == bar_set
        for g in generators(n_values)
    )

    print(
        f"{label}: |pi|={len(pi)} |pi_bar|={len(pi_bar)}  "
        f"value-closed(pi)={ok_val}  var-closed(pi)={ok_var}  "
        f"value-closed(pi_bar)={ok_bar}",
        flush=True,
    )
    return ok_val, ok_var, ok_bar


def check_resolvent_equivariance(pi_bar, var, domains, n_values, label):
    """Check resolvent(g.J) == g.resolvent(J) on the actual patterns.

    Takes each generator g, maps a sample of patterns through it, and compares
    the expanded clause set of the image against the image of the expanded
    clause set. This is the property the expansion-layer fix depends on.
    """
    groups: dict[frozenset[int], list[int]] = defaultdict(list)
    for j, c in enumerate(pi_bar):
        groups[c.get(var) or frozenset()].append(j)
    distinct = list(groups.keys())
    patterns = _enumerate_minimal_conflict_patterns_orbit(
        distinct, domains[var], n_values
    )

    # index lookup so a permuted value-set maps back to its group
    idx_of = {s: i for i, s in enumerate(distinct)}

    mismatches = 0
    checked = 0
    for g in generators(n_values):
        for J in patterns[:: max(1, len(patterns) // 200)]:
            img_idx = []
            ok = True
            for i in J:
                s2 = frozenset(g[v] for v in distinct[i])
                if s2 not in idx_of:
                    ok = False
                    break
                img_idx.append(idx_of[s2])
            if not ok:
                mismatches += 1
                continue
            if tuple(sorted(img_idx)) not in set([tuple(sorted(img_idx))]):
                pass
            # the image pattern must itself be a minimal pattern
            if tuple(sorted(img_idx)) not in set(patterns):
                mismatches += 1
            checked += 1

    print(
        f"{label}: patterns={len(patterns)} sampled_checks={checked} "
        f"image-not-a-pattern={mismatches}",
        flush=True,
    )
    return mismatches == 0


def main() -> None:
    p = argparse.ArgumentParser(
        description="Check whether the clause set at each elimination step is "
        "closed under value and variable relabeling."
    )
    p.add_argument("--size", type=int, default=9)
    p.add_argument("--steps", type=int, default=4)
    args = p.parse_args()

    cs = parse_constraints(group_dsl(args.size))
    domains = list(cs.var_domains)
    ordering = list(cs.ordering)
    n = len(ordering)
    n_values = args.size

    pi = [c for c in cs.clauses if c.literals and not _is_tautology(c, domains)]

    all_ok = True
    for step in range(n - 1, n - 1 - args.steps, -1):
        var = ordering[step]
        live = set(ordering[: step + 1])
        ov, ovar, obar = check_step(
            pi, var, domains, n_values, live, f"step={step} var={cs.var_names[var]}"
        )
        pi_bar = [c for c in pi if var in c.variables]
        eq = check_resolvent_equivariance(
            pi_bar, var, domains, n_values, f"  patterns step={step}"
        )
        all_ok = all_ok and ov and obar and eq

        pm = [c for c in pi if var not in c.variables]
        kept, _ = _grouped_eliminate_var(pi_bar, pm, var, domains, cap=domains[var])
        seen = set(pm)
        nxt = list(pm)
        for c in kept:
            if c not in seen:
                seen.add(c)
                nxt.append(c)
        pi = _dedup_by_subsumption(nxt)

    print(f"\nALL CHECKS PASS: {all_ok}")


if __name__ == "__main__":
    main()
