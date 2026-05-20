from __future__ import annotations

import json
import time
from dataclasses import dataclass
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Callable, Iterable

from signed_clauses import (
    ConstraintSet,
    SignedClause,
    SignedLiteral,
    clause_satisfied,
    parse_constraints,
)


class Unsatisfiable(Exception):
    """COMPILE produced the empty clause — the input is unsatisfiable."""


class CompileTooLarge(Exception):
    """Retained for backward compatibility. With cap = domain size, the cap is
    structurally tight (pigeonhole) and this is no longer raised by the
    standard compile path."""


@dataclass
class CompiledConstraints:
    var_names: list[str]
    var_domains: list[int]
    ordering: list[int]
    per_var_clauses: list[list[SignedClause]]


def _is_tautology(clause: SignedClause, var_domains: list[int]) -> bool:
    """A clause is tautological iff some literal's value-set is the full domain."""
    return any(
        len(lit.values) == var_domains[lit.var] for lit in clause.literals
    )


_HEARTBEAT_EVERY_K = 200_000


def _minimal_conflict_subsets(
    pi_bar: list[SignedClause],
    var: int,
    cap: int,
    verbose: bool = False,
    tag: str = "",
) -> list[tuple[int, ...]]:
    """Enumerate minimal J ⊆ [m] with ⋂_{j∈J} S_j[var] = ∅.

    Skips J whose proper subset already conflicts (subsumed J's yield
    redundant resolvents).

    `cap` is the largest |J| considered. With cap = var's domain size, this is
    structurally tight by pigeonhole: each clause added to a minimal J strictly
    shrinks the running intersection on var (else J wouldn't be minimal), and
    the intersection starts at size h and must reach 0, so |J| ≤ h.
    """
    m = len(pi_bar)
    s_sets = [pi_bar[j].get(var) or frozenset() for j in range(m)]
    minimal: list[tuple[int, ...]] = []
    max_size = min(m, cap)

    for size in range(1, max_size + 1):
        total = comb(m, size)
        if verbose:
            print(
                f"#   {tag}size={size} starting candidates={total} "
                f"cumulative_minimal={len(minimal)}",
                flush=True,
            )
        t_size = time.time()
        new_at_size = 0
        for idx, J in enumerate(combinations(range(m), size)):
            if verbose and idx and idx % _HEARTBEAT_EVERY_K == 0:
                elapsed = time.time() - t_size
                rate = idx / elapsed if elapsed > 0 else 0
                eta = (total - idx) / rate if rate > 0 else float("inf")
                print(
                    f"#     {tag}size={size} progress {idx}/{total} "
                    f"elapsed_s={elapsed:.1f} eta_s={eta:.1f} "
                    f"new_minimal_so_far={new_at_size}",
                    flush=True,
                )
            J_set = set(J)
            if any(set(prev).issubset(J_set) for prev in minimal):
                continue
            inter: frozenset[int] = s_sets[J[0]]
            for j in J[1:]:
                inter = inter & s_sets[j]
                if not inter:
                    break
            if not inter:
                minimal.append(J)
                new_at_size += 1
        if verbose:
            print(
                f"#   {tag}size={size} done new_minimal={new_at_size} "
                f"elapsed_s={time.time() - t_size:.3f} "
                f"cumulative_minimal={len(minimal)}",
                flush=True,
            )
    return minimal


def _disjoin(clauses: list[SignedClause]) -> SignedClause:
    """Disjunction of a list of clauses, merging duplicate variables (S1 ∪ S2)."""
    lits: list[SignedLiteral] = []
    for c in clauses:
        lits.extend(c.literals)
    return SignedClause.from_literals(lits)


def _subsumes(c1: SignedClause, c2: SignedClause) -> bool:
    """True iff c1 subsumes c2: every model of c1 is a model of c2.

    Equivalent to: vars(c1) ⊆ vars(c2), AND for every shared variable v,
    c1's value-set on v is a subset of c2's value-set on v.
    """
    if len(c1.literals) > len(c2.literals):
        return False
    c2_map = {lit.var: lit.values for lit in c2.literals}
    for lit in c1.literals:
        c2_vals = c2_map.get(lit.var)
        if c2_vals is None:
            return False
        if not lit.values.issubset(c2_vals):
            return False
    return True


def _dedup_by_subsumption(clauses: list[SignedClause]) -> list[SignedClause]:
    """Drop every clause that is subsumed by some other clause in the input.

    Sort smallest-first so the candidate subsumers are added before potential
    subsumees; then for each new clause check whether it subsumes any
    same-width clause already kept (smaller-width kept can't be subsumed
    since c subsumes k requires len(c) <= len(k)).
    """
    sorted_clauses = sorted(clauses, key=lambda c: len(c.literals))
    kept: list[SignedClause] = []
    for c in sorted_clauses:
        if any(_subsumes(k, c) for k in kept):
            continue
        if kept and len(kept[-1].literals) == len(c.literals):
            kept = [k for k in kept if not _subsumes(c, k)]
        kept.append(c)
    return kept


def compile_constraints(
    cs: ConstraintSet, verbose: bool = False
) -> CompiledConstraints:
    """Signed-resolution compile under cs.ordering.

    Raises:
        Unsatisfiable: if the empty clause is derived.
        CompileTooLarge: if a step would need too-large minimal subsets.
            The raised exception carries `step` and `cap` attributes.
    """
    domains = list(cs.var_domains)
    ordering = list(cs.ordering)
    n = len(ordering)

    pi_current: list[SignedClause] = [
        c for c in cs.clauses if c.literals and not _is_tautology(c, domains)
    ]

    preproc_before = len(pi_current)
    t_preproc = time.time()
    pi_current = _dedup_by_subsumption(pi_current)
    preproc_elapsed = time.time() - t_preproc
    preproc_eliminated = preproc_before - len(pi_current)

    per_var_clauses: list[list[SignedClause]] = [[] for _ in range(n)]

    if verbose:
        order_names = " ".join(cs.var_names[i] for i in ordering)
        print(
            f"# compile: n_vars={n} "
            f"input_clauses={len(cs.clauses)} "
            f"post_taut={preproc_before} "
            f"post_subsumption={len(pi_current)} "
            f"preproc_sub_elim={preproc_eliminated} "
            f"preproc_sub_elapsed_s={preproc_elapsed:.3f} "
            f"domains={sorted(set(domains))} "
            f"cap=per_var_domain",
            flush=True,
        )
        print(f"# ordering: {order_names}", flush=True)

    for step in range(n - 1, -1, -1):
        var = ordering[step]
        var_name = cs.var_names[var]
        t_step = time.time()

        pi_bar = [c for c in pi_current if var in c.variables]
        per_var_clauses[step] = pi_bar

        step_cap = domains[var]

        if verbose:
            print(
                f"# starting step={step} var={var_name} "
                f"pi_in={len(pi_bar)} pi_current={len(pi_current)} "
                f"cap={step_cap}",
                flush=True,
            )

        if not pi_bar:
            if verbose:
                print(
                    f"step={step} var={var_name} pi_in=0 "
                    f"n_minimal_J=0 n_resolvents_raw=0 n_resolvents_kept=0 "
                    f"max_J_size=0 pi_out_before_sub={len(pi_current)} "
                    f"pi_out_after_sub={len(pi_current)} sub_elim=0 "
                    f"sub_elapsed_s=0.000 "
                    f"elapsed_s={time.time() - t_step:.3f}",
                    flush=True,
                )
            continue

        try:
            minimal_Js = _minimal_conflict_subsets(
                pi_bar, var,
                cap=step_cap,
                verbose=verbose,
                tag=f"step={step} var={var_name} ",
            )
        except CompileTooLarge as e:
            e.step = step
            e.var_name = var_name
            e.cap = step_cap
            if verbose:
                print(
                    f"# CompileTooLarge at step={step} var={var_name} "
                    f"pi_in={len(pi_bar)} cap={step_cap} "
                    f"elapsed_s={time.time() - t_step:.3f}",
                    flush=True,
                )
            raise

        n_minimal_J = len(minimal_Js)
        max_J_size = max((len(J) for J in minimal_Js), default=0)

        conflict_clauses: list[SignedClause] = []
        for J in minimal_Js:
            disj = _disjoin([pi_bar[j].drop(var) for j in J])
            if not disj.literals:
                raise Unsatisfiable(
                    f"empty clause derived at variable "
                    f"{var_name!r} (step i={step + 1})"
                )
            if _is_tautology(disj, domains):
                continue
            conflict_clauses.append(disj)

        n_resolvents_raw = n_minimal_J

        pi_minus_bar = [c for c in pi_current if var not in c.variables]

        seen: set[SignedClause] = set(pi_minus_bar)
        next_pi: list[SignedClause] = list(pi_minus_bar)
        n_kept = 0
        for c in conflict_clauses:
            if c not in seen:
                seen.add(c)
                next_pi.append(c)
                n_kept += 1

        pi_out_before_sub = len(next_pi)
        t_sub = time.time()
        next_pi = _dedup_by_subsumption(next_pi)
        sub_elapsed = time.time() - t_sub
        pi_out_after_sub = len(next_pi)
        sub_eliminated = pi_out_before_sub - pi_out_after_sub

        pi_current = next_pi

        if verbose:
            print(
                f"step={step} var={var_name} pi_in={len(pi_bar)} "
                f"n_minimal_J={n_minimal_J} n_resolvents_raw={n_resolvents_raw} "
                f"n_resolvents_kept={n_kept} max_J_size={max_J_size} "
                f"pi_out_before_sub={pi_out_before_sub} "
                f"pi_out_after_sub={pi_out_after_sub} "
                f"sub_elim={sub_eliminated} sub_elapsed_s={sub_elapsed:.3f} "
                f"elapsed_s={time.time() - t_step:.3f}",
                flush=True,
            )

    assert all(c.literals for c in pi_current), \
        "post-compile residual contains an empty clause"
    assert all(not c.variables for c in pi_current), \
        "post-compile residual mentions variables in the ordering"

    return CompiledConstraints(
        var_names=list(cs.var_names),
        var_domains=list(cs.var_domains),
        ordering=list(cs.ordering),
        per_var_clauses=per_var_clauses,
    )


class ClauseFeasibility:
    """Feasibility backend that walks compiled per-variable clauses."""

    def __init__(self, compiled: CompiledConstraints):
        self.compiled = compiled
        self._neighbours = self._compute_neighbours()

    @classmethod
    def from_file(cls, path: str | Path) -> "ClauseFeasibility":
        with open(path) as fh:
            return cls(from_json(json.load(fh)))

    @property
    def var_domains(self) -> list[int]:
        return self.compiled.var_domains

    @property
    def ordering(self) -> list[int]:
        return self.compiled.ordering

    @property
    def neighbours(self) -> dict[int, frozenset[int]]:
        """Variable -> the set of other variables sharing some compiled clause."""
        return self._neighbours

    def _compute_neighbours(self) -> dict[int, frozenset[int]]:
        n = len(self.compiled.var_domains)
        accum: dict[int, set[int]] = {v: set() for v in range(n)}
        for pi_bar in self.compiled.per_var_clauses:
            for clause in pi_bar:
                vars_here = [lit.var for lit in clause.literals]
                for i, a in enumerate(vars_here):
                    for b in vars_here[i + 1:]:
                        accum[a].add(b)
                        accum[b].add(a)
        return {v: frozenset(s) for v, s in accum.items()}

    def feasible_set(
        self, step: int, prior_assignment: dict[int, int]
    ) -> frozenset[int]:
        """Compute S_i(a_<i) ⊆ [h_i] for variable ordering[step]."""
        var = self.compiled.ordering[step]
        domain = self.compiled.var_domains[var]
        pi_bar = self.compiled.per_var_clauses[step]

        feasible: frozenset[int] = frozenset(range(domain))
        for clause in pi_bar:
            phi_satisfied = False
            for lit in clause.literals:
                if lit.var == var:
                    continue
                if (
                    lit.var in prior_assignment
                    and prior_assignment[lit.var] in lit.values
                ):
                    phi_satisfied = True
                    break
            if phi_satisfied:
                continue
            s_j = clause.get(var) or frozenset()
            feasible = feasible & s_j
            if not feasible:
                break
        return feasible


def _clause_to_dsl(clause: SignedClause, var_names: list[str]) -> str:
    parts = []
    for lit in clause.literals:
        vs = ",".join(str(v) for v in sorted(lit.values))
        parts.append(f"[{vs}]:{var_names[lit.var]}")
    return " or ".join(parts)


def to_dsl(cc: CompiledConstraints) -> str:
    """Render a CompiledConstraints object as plain DSL text — re-parseable."""
    lines: list[str] = []
    for name, h in zip(cc.var_names, cc.var_domains):
        lines.append(f"var {name} {h}")
    lines.append("")
    lines.append("ordering " + " ".join(cc.var_names[i] for i in cc.ordering))
    lines.append("")

    for pi_bar in cc.per_var_clauses:
        for clause in pi_bar:
            lines.append(_clause_to_dsl(clause, cc.var_names))

    return "\n".join(lines).rstrip() + "\n"


def to_json(cc: CompiledConstraints) -> dict:
    return {
        "var_names": cc.var_names,
        "var_domains": cc.var_domains,
        "ordering": cc.ordering,
        "per_var_clauses": [
            [
                [[lit.var, sorted(lit.values)] for lit in c.literals]
                for c in pi_bar
            ]
            for pi_bar in cc.per_var_clauses
        ],
    }


def from_json(d: dict) -> CompiledConstraints:
    per_var_clauses = [
        [
            SignedClause(
                tuple(
                    SignedLiteral(int(v), frozenset(int(x) for x in vals))
                    for v, vals in clause
                )
            )
            for clause in pi_bar
        ]
        for pi_bar in d["per_var_clauses"]
    ]
    return CompiledConstraints(
        var_names=list(d["var_names"]),
        var_domains=[int(h) for h in d["var_domains"]],
        ordering=[int(i) for i in d["ordering"]],
        per_var_clauses=per_var_clauses,
    )


def enumerate_models(
    clauses: Iterable[SignedClause], var_domains: list[int]
) -> list[tuple[int, ...]]:
    """Brute-force enumerate every total assignment satisfying every clause."""
    n = len(var_domains)
    clauses = list(clauses)
    models: list[tuple[int, ...]] = []

    def rec(prefix: list[int]):
        if len(prefix) == n:
            assignment = {i: prefix[i] for i in range(n)}
            if all(clause_satisfied(c, assignment) for c in clauses):
                models.append(tuple(prefix))
            return
        for v in range(var_domains[len(prefix)]):
            prefix.append(v)
            rec(prefix)
            prefix.pop()

    rec([])
    return models


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Compile a signed-clause constraint file."
    )
    p.add_argument("input", type=Path, help="Input .txt with var/clause directives")
    p.add_argument(
        "output",
        type=Path,
        help="Output file. `.txt` -> plain DSL (default); `.json` -> structured dump.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step diagnostics (verbose is on by default for the CLI).",
    )
    args = p.parse_args()

    cs = parse_constraints(args.input.read_text())
    try:
        cc = compile_constraints(cs, verbose=not args.quiet)
    except Unsatisfiable as e:
        print(f"UNSAT: {e}", flush=True)
        raise SystemExit(1)
    except CompileTooLarge as e:
        step = getattr(e, "step", "?")
        var_name = getattr(e, "var_name", "?")
        cap = getattr(e, "cap", "?")
        print(
            f"TOO LARGE at step={step} var={var_name} cap={cap}: {e}",
            flush=True,
        )
        raise SystemExit(1)

    if args.output.suffix == ".json":
        args.output.write_text(json.dumps(to_json(cc), indent=2))
    else:
        args.output.write_text(to_dsl(cc))

    total = sum(len(c) for c in cc.per_var_clauses)
    n_res = sum(
        1 for pi_bar in cc.per_var_clauses
        for c in pi_bar
        if c not in set(cs.clauses)
    )
    print(
        f"Compiled {total} clause(s) "
        f"({total - n_res} originals + {n_res} resolvents) -> {args.output}"
    )


if __name__ == "__main__":
    _cli()
