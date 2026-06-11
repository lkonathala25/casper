from __future__ import annotations

import json
import os
import time
from collections import defaultdict
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
    value_base: int = 0


def _is_tautology(clause: SignedClause, var_domains: list[int]) -> bool:
    """A clause is tautological iff some literal's value-set is the full domain."""
    return any(
        len(lit.values) == var_domains[lit.var] for lit in clause.literals
    )


_HEARTBEAT_EVERY_K = 200_000


class _SubsumerIndex:
    """Inverted variable index for fast `does any kept clause subsume disj?` queries.

    Each subsumer is registered under exactly one pivot variable (the variable
    that currently has the smallest bucket — keeps buckets balanced). A query
    only walks subsumers whose pivot variable is mentioned by the candidate
    disj — which is necessary for subsumption (s.vars ⊆ disj.vars).

    Within each bucket, entries are ordered as inserted; the caller seeds the
    index with ambient clauses pre-sorted by ascending literal count so that
    short, strong subsumers are checked first.
    """

    __slots__ = ("buckets", "clause_maps", "ambient_count")

    def __init__(self, ambient: list[SignedClause]):
        self.buckets: dict[int, list[int]] = defaultdict(list)
        self.clause_maps: list[dict[int, frozenset[int]]] = []
        ambient_sorted = sorted(ambient, key=lambda c: len(c.literals))
        for c in ambient_sorted:
            self._insert(c)
        self.ambient_count = len(self.clause_maps)

    def _insert(self, clause: SignedClause) -> None:
        if not clause.literals:
            return
        cmap = {lit.var: lit.values for lit in clause.literals}
        pivot = min(cmap, key=lambda v: len(self.buckets[v]))
        idx = len(self.clause_maps)
        self.clause_maps.append(cmap)
        self.buckets[pivot].append(idx)

    def add_resolvent(self, clause: SignedClause) -> None:
        self._insert(clause)

    def find_subsumer(
        self, disj_map: dict[int, frozenset[int]], disj_len: int,
        exclude: int | None = None,
    ) -> int | None:
        """Return the index of any subsumer, or None. Index < ambient_count
        means the hit was from the seeded ambient pool. `exclude` skips one
        stored index (so a clause is not reported as subsuming itself)."""
        for v in disj_map:
            bucket = self.buckets.get(v)
            if not bucket:
                continue
            for idx in bucket:
                if idx == exclude:
                    continue
                cmap = self.clause_maps[idx]
                if len(cmap) > disj_len:
                    continue
                ok = True
                for s_var, s_vals in cmap.items():
                    d_vals = disj_map.get(s_var)
                    if d_vals is None or not s_vals.issubset(d_vals):
                        ok = False
                        break
                if ok:
                    return idx
        return None


def _enumerate_minimal_conflict_patterns(
    distinct_sets: list[frozenset[int]],
    cap: int,
    verbose: bool = False,
    tag: str = "",
) -> list[tuple[int, ...]]:
    """Enumerate minimal index-subsets of `distinct_sets` whose intersection is ∅.

    The ground set is the (at-most-2^h - 1) distinct value-sets on the variable
    being eliminated. By minimality, no two clauses with the same value-set can
    both belong to a minimal conflict J, so searching over *distinct* sets is
    equivalent to searching over clauses for the conflict-pattern question — but
    over a tiny universe (≤ 15 for h=4) instead of a large one (|pi_bar|).
    """
    m = len(distinct_sets)
    minimal: list[tuple[int, ...]] = []
    max_size = min(m, cap)

    for size in range(1, max_size + 1):
        total = comb(m, size)
        t_size = time.time()
        new_at_size = 0
        for J in combinations(range(m), size):
            J_set = set(J)
            if any(set(prev).issubset(J_set) for prev in minimal):
                continue
            inter: frozenset[int] = distinct_sets[J[0]]
            for j in J[1:]:
                inter = inter & distinct_sets[j]
                if not inter:
                    break
            if not inter:
                minimal.append(J)
                new_at_size += 1
        if verbose:
            print(
                f"#   {tag}pattern-size={size} candidates={total} "
                f"new_minimal={new_at_size} "
                f"cumulative_minimal={len(minimal)} "
                f"elapsed_s={time.time() - t_size:.3f}",
                flush=True,
            )
    return minimal


_PAIRWISE_SIZE4_THRESHOLD = 1_000_000

# Reorder the groups of a conflict pattern so that the groups whose clauses
# carry the widest value-sets (most likely to saturate some variable's domain)
# are expanded first. Prefix-tautology pruning then fires at a shallower depth
# and discards a larger subtree. This is a pure permutation of the per-depth
# group assignment: a resolvent depends only on the *set* of picked clauses
# (union is order-independent), so the leaf clause set — and hence the compiled
# output after subsumption — is unchanged. Env override SAT_REORDER_GROUPS=0/1
# toggles it for A/B measurement.
_REORDER_GROUPS_BY_SATURATION = os.environ.get("SAT_REORDER_GROUPS", "1") != "0"

# Minimum pile size before the per-step subsumption cleanup is parallelized;
# below this, pool startup costs more than the serial pass saves.
_PARALLEL_DEDUP_MIN = 5_000


def _pairwise_stage(
    g_a_indices: list[int],
    g_b_indices: list[int],
    pi_bar: list[SignedClause],
    var: int,
    domains: list[int],
    ambient_subsumers: list[SignedClause],
    verbose: bool = False,
    tag: str = "",
    label: str = "",
) -> tuple[list[SignedClause], dict]:
    """Compute distinct pairwise resolvents from g_a × g_b for one stage of the
    size-4 pairwise expansion. Each result clause merges the literals of one
    clause from each group; the var literal is kept with value-set S_a ∪ S_b
    unless that union is the full domain (in which case it's dropped).

    Forward subsumption is applied against `ambient_subsumers` and against
    earlier intermediates in this stage. Tautologies and duplicates are skipped.
    """
    full_domain = domains[var]
    intermediates: list[SignedClause] = []
    seen: set[SignedClause] = set()
    index = _SubsumerIndex(ambient_subsumers)

    n_raw = 0
    n_taut = 0
    n_dup = 0
    n_subsumed = 0
    t_start = time.time()
    last_hb = t_start

    for j_a in g_a_indices:
        c_a = pi_bar[j_a]
        s_a = c_a.get(var) or frozenset()
        a_lits: dict[int, frozenset[int]] = {}
        for lit in c_a.literals:
            if lit.var != var:
                a_lits[lit.var] = lit.values

        for j_b in g_b_indices:
            c_b = pi_bar[j_b]
            s_b = c_b.get(var) or frozenset()
            n_raw += 1

            merged = dict(a_lits)
            for lit in c_b.literals:
                if lit.var == var:
                    continue
                existing = merged.get(lit.var)
                merged[lit.var] = (existing | lit.values) if existing else lit.values

            y_union = s_a | s_b
            if y_union and len(y_union) < full_domain:
                merged[var] = y_union

            if any(len(vals) == domains[v] for v, vals in merged.items()):
                n_taut += 1
            elif not merged:
                raise Unsatisfiable(
                    f"empty clause derived at variable index {var} "
                    f"during pairwise stage-{label}"
                )
            else:
                literals = tuple(
                    SignedLiteral(v, vals) for v, vals in sorted(merged.items())
                )
                clause = SignedClause(literals)
                if clause in seen:
                    n_dup += 1
                else:
                    hit_idx = index.find_subsumer(merged, len(literals))
                    if hit_idx is not None:
                        n_subsumed += 1
                    else:
                        intermediates.append(clause)
                        seen.add(clause)
                        index.add_resolvent(clause)

            if verbose and n_raw % _HEARTBEAT_EVERY_K == 0:
                now = time.time()
                rate = _HEARTBEAT_EVERY_K / max(now - last_hb, 1e-9)
                last_hb = now
                print(
                    f"#     {tag}stage-{label} progress raw={n_raw} "
                    f"kept={len(intermediates)} taut={n_taut} dup={n_dup} "
                    f"sub={n_subsumed} rate={rate:.0f}/s",
                    flush=True,
                )

    elapsed = time.time() - t_start
    if verbose:
        print(
            f"#   {tag}stage-{label} done raw={n_raw} "
            f"intermediates={len(intermediates)} taut={n_taut} "
            f"dup={n_dup} sub={n_subsumed} elapsed_s={elapsed:.3f}",
            flush=True,
        )

    return intermediates, {
        "raw": n_raw,
        "kept": len(intermediates),
        "taut": n_taut,
        "dup": n_dup,
        "subsumed": n_subsumed,
        "elapsed": elapsed,
    }


def _grouped_eliminate_var(
    pi_bar: list[SignedClause],
    pi_minus_bar: list[SignedClause],
    var: int,
    domains: list[int],
    cap: int,
    verbose: bool = False,
    tag: str = "",
    only_pattern_size: int | None = None,
    work_pattern_idx: int | None = None,
    work_first_chunk: tuple[int, int] | None = None,
    dump_raw: list | None = None,
) -> tuple[list[SignedClause], dict]:
    """Eliminate `var` by grouping pi_bar by value-set, enumerating minimal
    conflict patterns over distinct value-sets, then expanding each pattern
    via cross-product with forward-subsumption pruning.

    Returns (kept_resolvents, stats).
    """
    groups: dict[frozenset[int], list[int]] = defaultdict(list)
    for j, c in enumerate(pi_bar):
        s = c.get(var) or frozenset()
        groups[s].append(j)

    distinct_sets = list(groups.keys())
    n_groups = len(distinct_sets)

    if verbose:
        avg = len(pi_bar) / n_groups if n_groups else 0.0
        print(
            f"#   {tag}grouped pi_bar={len(pi_bar)} "
            f"distinct_value_sets={n_groups} "
            f"avg_group_size={avg:.1f}",
            flush=True,
        )

    patterns = _enumerate_minimal_conflict_patterns(
        distinct_sets, cap, verbose=verbose, tag=tag
    )

    # Fan the per-pattern expansion across worker processes when requested
    # (SAT_WORKERS>1). Workers re-enter this function with a fixed work item
    # (work_pattern_idx set), so they take the serial path below — no recursion.
    n_workers = int(os.environ.get("SAT_WORKERS", "1"))
    if n_workers > 1 and work_pattern_idx is None:
        return _grouped_eliminate_parallel(
            pi_bar, pi_minus_bar, var, domains, cap, patterns, groups,
            distinct_sets, n_workers, verbose, tag, only_pattern_size,
        )

    pi_bar_dropped = [c.drop(var) for c in pi_bar]
    index = _SubsumerIndex(pi_minus_bar)
    ambient_count = index.ambient_count

    kept: list[SignedClause] = []
    seen: set[SignedClause] = set(pi_minus_bar)
    counters = {
        "n_raw": 0,
        "n_taut": 0,
        "n_duplicate": 0,
        "n_subsumed_ambient": 0,
        "n_subsumed_intra": 0,
        "n_prefix_taut": 0,
        "n_lookahead_taut": 0,
        "n_prefix_subsumed": 0,
        "n_pruned_leaves": 0,
    }
    t_expand = time.time()
    hb_state = {"last_t": t_expand}

    def maybe_heartbeat() -> None:
        if not verbose:
            return
        n = counters["n_raw"]
        if n and n % _HEARTBEAT_EVERY_K == 0:
            now = time.time()
            rate = _HEARTBEAT_EVERY_K / max(now - hb_state["last_t"], 1e-9)
            hb_state["last_t"] = now
            print(
                f"#     {tag}expand progress raw={n} kept={len(kept)} "
                f"taut={counters['n_taut']} dup={counters['n_duplicate']} "
                f"sub_amb={counters['n_subsumed_ambient']} "
                f"sub_intra={counters['n_subsumed_intra']} "
                f"prefix_taut={counters['n_prefix_taut']} "
                f"lookahead_taut={counters['n_lookahead_taut']} "
                f"prefix_sub={counters['n_prefix_subsumed']} "
                f"pruned_leaves={counters['n_pruned_leaves']} "
                f"rate={rate:.0f}/s",
                flush=True,
            )

    def expand(group_lists: list[list[int]]) -> None:
        # Front-load the groups most likely to saturate a variable so that
        # prefix-tautology pruning fires at a shallower depth (pruning a larger
        # subtree). Primary key: descending average clause width (sum of
        # value-set sizes per clause) — wider clauses fill domains faster.
        # Secondary key: ascending group size, so the largest groups sit at the
        # deepest levels and a shallow prune discards the most leaves. Pure
        # permutation of depth assignment; the leaf clause set is invariant.
        if _REORDER_GROUPS_BY_SATURATION and len(group_lists) > 2:
            def _saturation_key(gl: list[int]) -> tuple[float, int]:
                if not gl:
                    return (0.0, 0)
                width = sum(
                    len(lit.values)
                    for j in gl
                    for lit in pi_bar_dropped[j].literals
                )
                return (-width / len(gl), len(gl))
            group_lists = sorted(group_lists, key=_saturation_key)

        pattern_size = len(group_lists)
        # Per-depth descendant leaf counts (for accounting pruned subtrees).
        suffix_sizes = [1] * (pattern_size + 1)
        for d in range(pattern_size - 1, -1, -1):
            suffix_sizes[d] = suffix_sizes[d + 1] * len(group_lists[d])

        # Per-group FORCED contributions per variable. For each group g and
        # var v, the set of values that *every* clause in g contributes — i.e.
        # the intersection of V_c[v] across c in g (with ∅ if any clause in g
        # doesn't mention v at all, since we could pick that clause).
        # These are values we cannot avoid adding to v regardless of choice.
        forced_contrib_per_group: list[dict[int, frozenset[int]]] = []
        for gl in group_lists:
            contrib: dict[int, frozenset[int]] = {}
            all_vars: set[int] = set()
            for j in gl:
                for lit in pi_bar_dropped[j].literals:
                    all_vars.add(lit.var)
            for v in all_vars:
                inter: frozenset[int] | None = None
                missing = False
                for j in gl:
                    cv = pi_bar_dropped[j].get(v)
                    if cv is None:
                        missing = True
                        break
                    inter = cv if inter is None else (inter & cv)
                    if not inter:
                        break
                if missing or inter is None or not inter:
                    contrib[v] = frozenset()
                else:
                    contrib[v] = inter
            forced_contrib_per_group.append(contrib)

        # tail[d][v] = union over groups i in [d+1..pattern_size-1] of
        # forced_contrib_per_group[i].get(v, ∅). The set of values that
        # remaining groups will *force* to be added to v on every leaf path.
        tail: list[dict[int, frozenset[int]]] = [dict() for _ in range(pattern_size)]
        for d in range(pattern_size - 2, -1, -1):
            merged: dict[int, frozenset[int]] = dict(tail[d + 1])
            for v, vs in forced_contrib_per_group[d + 1].items():
                if not vs:
                    continue
                existing = merged.get(v)
                merged[v] = (existing | vs) if existing else vs
            tail[d] = merged

        def lookahead_taut(
            child_map: dict[int, frozenset[int]], depth: int
        ) -> bool:
            tail_d = tail[depth]
            if not tail_d:
                return False
            for v, vs in child_map.items():
                tv = tail_d.get(v)
                best = (vs | tv) if tv else vs
                if len(best) == domains[v]:
                    return True
            for v, tv in tail_d.items():
                if v in child_map:
                    continue
                if len(tv) == domains[v]:
                    return True
            return False

        def recurse(depth: int, cur_map: dict[int, frozenset[int]]) -> None:
            if depth == pattern_size:
                counters["n_raw"] += 1
                if not cur_map:
                    raise Unsatisfiable(
                        f"empty clause derived at variable index {var} "
                        f"during grouped expansion"
                    )
                literals = tuple(
                    SignedLiteral(v, vals)
                    for v, vals in sorted(cur_map.items())
                )
                disj = SignedClause(literals)
                if any(len(vals) == domains[v] for v, vals in cur_map.items()):
                    counters["n_taut"] += 1
                    if dump_raw is not None:
                        dump_raw.append(("tautology", disj))
                    maybe_heartbeat()
                    return
                if disj in seen:
                    counters["n_duplicate"] += 1
                    if dump_raw is not None:
                        dump_raw.append(("duplicate", disj))
                    maybe_heartbeat()
                    return
                hit_idx = index.find_subsumer(cur_map, len(literals))
                if hit_idx is not None:
                    if hit_idx < ambient_count:
                        counters["n_subsumed_ambient"] += 1
                    else:
                        counters["n_subsumed_intra"] += 1
                    if dump_raw is not None:
                        dump_raw.append(("subsumed", disj))
                    maybe_heartbeat()
                    return
                kept.append(disj)
                seen.add(disj)
                index.add_resolvent(disj)
                if dump_raw is not None:
                    dump_raw.append(("kept", disj))
                maybe_heartbeat()
                return

            for j in group_lists[depth]:
                child_map = dict(cur_map)
                # Build the union one literal at a time and STOP the moment any
                # variable fills its whole domain: that makes the resolvent a
                # tautology, so there's no point finishing it or recursing. This
                # catches tautologies at every depth — including the last, where
                # the leaf would otherwise build the full clause just to discard
                # it. (When dumping raw resolvents we finish the union so the
                # discarded clause can still be recorded.)
                saturated = False
                for lit in pi_bar_dropped[j].literals:
                    existing = child_map.get(lit.var)
                    merged = (existing | lit.values) if existing else lit.values
                    child_map[lit.var] = merged
                    if len(merged) == domains[lit.var]:
                        saturated = True
                        if dump_raw is None:
                            break
                if saturated:
                    if depth + 1 < pattern_size:
                        counters["n_prefix_taut"] += 1
                        counters["n_pruned_leaves"] += suffix_sizes[depth + 1]
                    else:
                        counters["n_raw"] += 1
                        counters["n_taut"] += 1
                        if dump_raw is not None:
                            dump_raw.append(("tautology", SignedClause(tuple(
                                SignedLiteral(v, vals)
                                for v, vals in sorted(child_map.items())))))
                    maybe_heartbeat()
                    continue

                if depth + 1 < pattern_size and child_map:
                    if lookahead_taut(child_map, depth):
                        counters["n_lookahead_taut"] += 1
                        counters["n_pruned_leaves"] += suffix_sizes[depth + 1]
                        continue
                    hit = index.find_subsumer(child_map, len(child_map))
                    if hit is not None:
                        counters["n_prefix_subsumed"] += 1
                        counters["n_pruned_leaves"] += suffix_sizes[depth + 1]
                        continue

                recurse(depth + 1, child_map)

        recurse(0, {})

    def expand_pairwise(group_lists: list[list[int]]) -> None:
        """Size-4 pattern: two pairwise stages instead of one 4-deep cross-product."""
        full_domain = domains[var]

        i01, stats_1a = _pairwise_stage(
            group_lists[0], group_lists[1], pi_bar, var, domains,
            ambient_subsumers=pi_minus_bar,
            verbose=verbose, tag=tag, label="1a",
        )
        i23, stats_1b = _pairwise_stage(
            group_lists[2], group_lists[3], pi_bar, var, domains,
            ambient_subsumers=pi_minus_bar,
            verbose=verbose, tag=tag, label="1b",
        )

        t_stage2 = time.time()
        n_stage2_raw_start = counters["n_raw"]

        for c01 in i01:
            a_lits: dict[int, frozenset[int]] = {}
            a_y: frozenset[int] = frozenset()
            for lit in c01.literals:
                if lit.var == var:
                    a_y = lit.values
                else:
                    a_lits[lit.var] = lit.values

            for c23 in i23:
                counters["n_raw"] += 1

                merged = dict(a_lits)
                b_y: frozenset[int] = frozenset()
                for lit in c23.literals:
                    if lit.var == var:
                        b_y = lit.values
                        continue
                    existing = merged.get(lit.var)
                    merged[lit.var] = (existing | lit.values) if existing else lit.values

                y_union = a_y | b_y
                if y_union and len(y_union) < full_domain:
                    merged[var] = y_union

                if any(len(vals) == domains[v] for v, vals in merged.items()):
                    counters["n_taut"] += 1
                    maybe_heartbeat()
                    continue

                if not merged:
                    raise Unsatisfiable(
                        f"empty clause derived at variable index {var} "
                        f"during pairwise stage-2"
                    )

                literals = tuple(
                    SignedLiteral(v, vals) for v, vals in sorted(merged.items())
                )
                disj = SignedClause(literals)
                if disj in seen:
                    counters["n_duplicate"] += 1
                    maybe_heartbeat()
                    continue
                hit_idx = index.find_subsumer(merged, len(literals))
                if hit_idx is not None:
                    if hit_idx < ambient_count:
                        counters["n_subsumed_ambient"] += 1
                    else:
                        counters["n_subsumed_intra"] += 1
                    maybe_heartbeat()
                    continue
                kept.append(disj)
                seen.add(disj)
                index.add_resolvent(disj)
                maybe_heartbeat()

        if verbose:
            stage2_raw = counters["n_raw"] - n_stage2_raw_start
            print(
                f"#   {tag}stage-2 done raw={stage2_raw} "
                f"intermediates_01={len(i01)} intermediates_23={len(i23)} "
                f"cumulative_kept={len(kept)} "
                f"elapsed_s={time.time() - t_stage2:.3f}",
                flush=True,
            )

    for p_idx, pattern in enumerate(patterns):
        if only_pattern_size is not None and len(pattern) != only_pattern_size:
            continue
        if work_pattern_idx is not None and p_idx != work_pattern_idx:
            continue
        group_lists = [groups[distinct_sets[i]] for i in pattern]
        expansion_size = 1
        for gl in group_lists:
            expansion_size *= len(gl)
        use_pairwise = (
            len(pattern) == 4 and expansion_size > _PAIRWISE_SIZE4_THRESHOLD
        )
        # For a parallel work item, restrict the first group to its assigned
        # chunk. The pairwise decision above is computed from the *full* pattern
        # first, so chunking never reroutes a huge pattern into the 4-deep
        # recursion. The union of all chunks reproduces the full pattern's
        # resolvents (cross-chunk duplicates removed by the caller's dedup).
        if work_first_chunk is not None:
            cs0, ce0 = work_first_chunk
            group_lists = [group_lists[0][cs0:ce0]] + group_lists[1:]
        if verbose:
            mode = " mode=pairwise" if use_pairwise else ""
            print(
                f"#   {tag}pattern {p_idx + 1}/{len(patterns)} "
                f"size={len(pattern)} expansion={expansion_size} "
                f"cumulative_kept={len(kept)}{mode}",
                flush=True,
            )
        if use_pairwise:
            expand_pairwise(group_lists)
        else:
            expand(group_lists)

    stats = {
        "n_distinct_sets": n_groups,
        "n_patterns": len(patterns),
        "max_pattern_size": max((len(p) for p in patterns), default=0),
        "n_resolvents_raw": counters["n_raw"],
        "n_tautologies": counters["n_taut"],
        "n_duplicates": counters["n_duplicate"],
        "n_fwd_subsumed_ambient": counters["n_subsumed_ambient"],
        "n_fwd_subsumed_intra": counters["n_subsumed_intra"],
        "n_prefix_taut": counters["n_prefix_taut"],
        "n_lookahead_taut": counters["n_lookahead_taut"],
        "n_prefix_subsumed": counters["n_prefix_subsumed"],
        "n_pruned_leaves": counters["n_pruned_leaves"],
        "n_kept": len(kept),
        "expand_elapsed_s": time.time() - t_expand,
    }
    return kept, stats


# Read-only inputs shared with worker processes via fork inheritance (avoids
# pickling the large clause lists per task). Set by _grouped_eliminate_parallel
# in the parent immediately before the worker pool is created.
_PARALLEL_CTX: dict = {}


def _parallel_task(item: tuple[int, tuple[int, int]]):
    """Worker: expand one conflict pattern (a first-group chunk of it) by
    re-entering the serial machinery with a fixed work item. Big read-only
    inputs come from the inherited _PARALLEL_CTX so only the tiny (pattern,
    chunk) tuple is pickled."""
    p_idx, chunk = item
    ctx = _PARALLEL_CTX
    return _grouped_eliminate_var(
        ctx["pi_bar"], ctx["pi_minus_bar"], ctx["var"], ctx["domains"],
        ctx["cap"], verbose=False,
        work_pattern_idx=p_idx, work_first_chunk=chunk,
    )


def _grouped_eliminate_parallel(
    pi_bar: list[SignedClause],
    pi_minus_bar: list[SignedClause],
    var: int,
    domains: list[int],
    cap: int,
    patterns: list[tuple[int, ...]],
    groups: dict[frozenset[int], list[int]],
    distinct_sets: list[frozenset[int]],
    n_workers: int,
    verbose: bool,
    tag: str,
    only_pattern_size: int | None,
) -> tuple[list[SignedClause], dict]:
    """Parallel per-pattern expansion: split each pattern's first group into
    chunks and expand the chunks across a process pool. Each worker subsumes
    only against the ambient set, so cross-chunk duplicates/subsumees are left
    for the caller's final subsumption dedup; the eliminated variable's
    resolvent set is identical to the serial result."""
    import multiprocessing as mp

    tasks_per_worker = 4
    items: list[tuple[int, tuple[int, int]]] = []
    for p_idx, pattern in enumerate(patterns):
        if only_pattern_size is not None and len(pattern) != only_pattern_size:
            continue
        n0 = len(groups[distinct_sets[pattern[0]]])
        if n0 == 0:
            items.append((p_idx, (0, 0)))
            continue
        target = max(1, n_workers * tasks_per_worker)
        step = max(1, (n0 + target - 1) // target)
        for s in range(0, n0, step):
            items.append((p_idx, (s, min(s + step, n0))))

    global _PARALLEL_CTX
    _PARALLEL_CTX = {
        "pi_bar": pi_bar, "pi_minus_bar": pi_minus_bar, "var": var,
        "domains": domains, "cap": cap,
    }

    if verbose:
        print(
            f"#   {tag}parallel workers={n_workers} tasks={len(items)}",
            flush=True,
        )

    t_expand = time.time()
    with mp.get_context("fork").Pool(n_workers) as pool:
        results = pool.map(_parallel_task, items)
    pool_elapsed = time.time() - t_expand

    kept: list[SignedClause] = []
    seen: set[SignedClause] = set(pi_minus_bar)
    agg = {
        "n_resolvents_raw": 0, "n_tautologies": 0, "n_duplicates": 0,
        "n_fwd_subsumed_ambient": 0, "n_fwd_subsumed_intra": 0,
        "n_prefix_taut": 0, "n_lookahead_taut": 0, "n_prefix_subsumed": 0,
        "n_pruned_leaves": 0,
    }
    for kpart, spart in results:
        for c in kpart:
            if c not in seen:
                seen.add(c)
                kept.append(c)
        for k in agg:
            agg[k] += spart.get(k, 0)

    if verbose:
        print(
            f"#   {tag}parallel done kept={len(kept)} "
            f"raw={agg['n_resolvents_raw']} taut={agg['n_tautologies']} "
            f"elapsed_s={pool_elapsed:.3f}",
            flush=True,
        )

    stats = {
        "n_distinct_sets": len(distinct_sets),
        "n_patterns": len(patterns),
        "max_pattern_size": max((len(p) for p in patterns), default=0),
        "n_kept": len(kept),
        "expand_elapsed_s": pool_elapsed,
        **agg,
    }
    return kept, stats


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


# Read-only state shared with dedup workers via fork (set in the parent right
# before the pool is created). The subsumer index is queried, never mutated.
_DEDUP_CTX: dict = {}


def _dedup_slice(bounds: tuple[int, int]) -> list[SignedClause]:
    """Worker: keep clauses in [lo, hi) that no *other* clause subsumes."""
    lo, hi = bounds
    index = _DEDUP_CTX["index"]
    maps = _DEDUP_CTX["maps"]
    clauses = _DEDUP_CTX["clauses"]
    out: list[SignedClause] = []
    for i in range(lo, hi):
        if index.find_subsumer(maps[i], len(maps[i]), exclude=i) is None:
            out.append(clauses[i])
    return out


def _dedup_by_subsumption_parallel(
    clauses: list[SignedClause], n_workers: int
) -> list[SignedClause]:
    """Parallel equivalent of `_dedup_by_subsumption`: the subsumption-minimal
    subset (every clause not subsumed by some other). Each clause's "is anything
    else subsuming me?" test is independent, so they fan out across workers.
    Returns the same set as the serial version (order may differ, which the
    compiled constraint set is invariant to)."""
    import multiprocessing as mp

    unique = list(dict.fromkeys(clauses))  # collapse exact duplicates, keep order
    # _SubsumerIndex sorts by length on insert; pre-sort so `maps[i]` aligns
    # with `unique_sorted[i]` (stable sort leaves the pre-sorted order intact).
    unique_sorted = sorted(unique, key=lambda c: len(c.literals))
    index = _SubsumerIndex(unique_sorted)
    maps = index.clause_maps

    global _DEDUP_CTX
    _DEDUP_CTX = {"index": index, "maps": maps, "clauses": unique_sorted}

    n = len(unique_sorted)
    step = max(1, (n + n_workers * 4 - 1) // (n_workers * 4))
    bounds = [(s, min(s + step, n)) for s in range(0, n, step)]
    with mp.get_context("fork").Pool(n_workers) as pool:
        parts = pool.map(_dedup_slice, bounds)
    return [c for part in parts for c in part]


def _primal_adjacency(cs: ConstraintSet) -> list[set[int]]:
    """Primal constraint graph: edge u-v iff some clause mentions both."""
    n = len(cs.var_domains)
    adj: list[set[int]] = [set() for _ in range(n)]
    for clause in cs.clauses:
        vars_here = [lit.var for lit in clause.literals]
        for i, a in enumerate(vars_here):
            for b in vars_here[i + 1:]:
                if a != b:
                    adj[a].add(b)
                    adj[b].add(a)
    return adj


def _heuristic_elimination_ordering(
    cs: ConstraintSet, heuristic: str = "min-fill"
) -> list[int]:
    """Compute an elimination ordering by simulating elimination on the primal
    graph. `heuristic` is "min-degree" or "min-fill".

    Returns an `ordering` list compatible with compile_constraints: the variable
    at index n-1 is eliminated first (compile iterates step from n-1 down to 0).
    """
    n = len(cs.var_domains)
    adj = _primal_adjacency(cs)
    remaining = set(range(n))
    elim_order: list[int] = []

    def fill_in(v: int) -> int:
        ns = adj[v] & remaining
        nlist = list(ns)
        count = 0
        for i, a in enumerate(nlist):
            for b in nlist[i + 1:]:
                if b not in adj[a]:
                    count += 1
        return count

    while remaining:
        if heuristic == "min-degree":
            chosen = min(remaining, key=lambda x: len(adj[x] & remaining))
        elif heuristic == "min-fill":
            chosen = min(remaining, key=lambda x: (fill_in(x), len(adj[x] & remaining)))
        else:
            raise ValueError(f"unknown heuristic {heuristic!r}")
        elim_order.append(chosen)
        ns = adj[chosen] & remaining
        for a in ns:
            for b in ns:
                if a != b:
                    adj[a].add(b)
        remaining.remove(chosen)

    return list(reversed(elim_order))


def compile_constraints(
    cs: ConstraintSet,
    verbose: bool = False,
    legacy_enum: bool = False,
    stream_to: "Path | str | None" = None,
) -> CompiledConstraints:
    """Signed-resolution compile under cs.ordering.

    By default uses the grouped + forward-subsumption enumeration; set
    `legacy_enum=True` to fall back to the original per-clause combinations
    enumeration (kept for A/B comparison).

    If `stream_to` is given, each step's finalised pi_bar is appended to that
    file as DSL lines (with `# step=N var=NAME` header comments) and flushed.
    Useful for live observability and partial-artifact survival under SIGINT.
    The file is re-parseable (comments are ignored by parse_constraints).

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

    stream_file = None
    if stream_to is not None:
        stream_file = open(stream_to, "w")
        for name, h in zip(cs.var_names, cs.var_domains):
            stream_file.write(f"var {name} {h}\n")
        if cs.value_base:
            stream_file.write(f"\nvalue-base {cs.value_base}\n")
        stream_file.write("\n")
        stream_file.write(
            "ordering " + " ".join(cs.var_names[i] for i in ordering) + "\n"
        )
        stream_file.write("\n")
        stream_file.write("# streaming: clauses appear in elimination order\n")
        stream_file.flush()

    if verbose:
        order_names = " ".join(cs.var_names[i] for i in ordering)
        algo = "legacy" if legacy_enum else "grouped+fwdsub"
        print(
            f"# compile: n_vars={n} "
            f"input_clauses={len(cs.clauses)} "
            f"post_taut={preproc_before} "
            f"post_subsumption={len(pi_current)} "
            f"preproc_sub_elim={preproc_eliminated} "
            f"preproc_sub_elapsed_s={preproc_elapsed:.3f} "
            f"domains={sorted(set(domains))} "
            f"cap=per_var_domain algo={algo}",
            flush=True,
        )
        print(f"# ordering: {order_names}", flush=True)

    for step in range(n - 1, -1, -1):
        var = ordering[step]
        var_name = cs.var_names[var]
        t_step = time.time()

        pi_bar = [c for c in pi_current if var in c.variables]
        per_var_clauses[step] = pi_bar

        if stream_file is not None:
            stream_file.write(
                f"\n# step={step} var={var_name} ({len(pi_bar)} clauses)\n"
            )
            for clause in pi_bar:
                stream_file.write(
                    _clause_to_dsl(clause, cs.var_names, cs.value_base) + "\n"
                )
            stream_file.flush()

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

        pi_minus_bar = [c for c in pi_current if var not in c.variables]

        if legacy_enum:
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

            kept_resolvents: list[SignedClause] = []
            for J in minimal_Js:
                disj = _disjoin([pi_bar[j].drop(var) for j in J])
                if not disj.literals:
                    raise Unsatisfiable(
                        f"empty clause derived at variable "
                        f"{var_name!r} (step i={step + 1})"
                    )
                if _is_tautology(disj, domains):
                    continue
                kept_resolvents.append(disj)

            n_resolvents_raw = n_minimal_J
            grouped_stats: dict | None = None
        else:
            kept_resolvents, grouped_stats = _grouped_eliminate_var(
                pi_bar, pi_minus_bar, var, domains,
                cap=step_cap,
                verbose=verbose,
                tag=f"step={step} var={var_name} ",
            )
            n_minimal_J = grouped_stats["n_patterns"]
            max_J_size = grouped_stats["max_pattern_size"]
            n_resolvents_raw = grouped_stats["n_resolvents_raw"]

        seen: set[SignedClause] = set(pi_minus_bar)
        next_pi: list[SignedClause] = list(pi_minus_bar)
        n_kept = 0
        for c in kept_resolvents:
            if c not in seen:
                seen.add(c)
                next_pi.append(c)
                n_kept += 1

        pi_out_before_sub = len(next_pi)
        t_sub = time.time()
        # The final subsumption cleanup is the single-threaded tail of a
        # parallel step; fan it across workers too once the pile is big enough
        # to outweigh pool overhead. Identical result, order may differ.
        sub_workers = int(os.environ.get("SAT_WORKERS", "1"))
        if sub_workers > 1 and len(next_pi) > _PARALLEL_DEDUP_MIN:
            next_pi = _dedup_by_subsumption_parallel(next_pi, sub_workers)
        else:
            next_pi = _dedup_by_subsumption(next_pi)
        sub_elapsed = time.time() - t_sub
        pi_out_after_sub = len(next_pi)
        sub_eliminated = pi_out_before_sub - pi_out_after_sub

        pi_current = next_pi

        if verbose:
            extra = ""
            if grouped_stats is not None:
                extra = (
                    f" n_distinct_sets={grouped_stats['n_distinct_sets']}"
                    f" n_taut={grouped_stats['n_tautologies']}"
                    f" fwd_sub_amb={grouped_stats['n_fwd_subsumed_ambient']}"
                    f" fwd_sub_intra={grouped_stats['n_fwd_subsumed_intra']}"
                    f" prefix_taut={grouped_stats['n_prefix_taut']}"
                    f" lookahead_taut={grouped_stats['n_lookahead_taut']}"
                    f" prefix_sub={grouped_stats['n_prefix_subsumed']}"
                    f" pruned_leaves={grouped_stats['n_pruned_leaves']}"
                    f" dup={grouped_stats['n_duplicates']}"
                    f" expand_elapsed_s={grouped_stats['expand_elapsed_s']:.3f}"
                )
            print(
                f"step={step} var={var_name} pi_in={len(pi_bar)} "
                f"n_minimal_J={n_minimal_J} n_resolvents_raw={n_resolvents_raw} "
                f"n_resolvents_kept={n_kept} max_J_size={max_J_size} "
                f"pi_out_before_sub={pi_out_before_sub} "
                f"pi_out_after_sub={pi_out_after_sub} "
                f"sub_elim={sub_eliminated} sub_elapsed_s={sub_elapsed:.3f}"
                f"{extra} "
                f"elapsed_s={time.time() - t_step:.3f}",
                flush=True,
            )

    assert all(c.literals for c in pi_current), \
        "post-compile residual contains an empty clause"
    assert all(not c.variables for c in pi_current), \
        "post-compile residual mentions variables in the ordering"

    if stream_file is not None:
        stream_file.close()

    return CompiledConstraints(
        var_names=list(cs.var_names),
        var_domains=list(cs.var_domains),
        ordering=list(cs.ordering),
        per_var_clauses=per_var_clauses,
        value_base=cs.value_base,
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


def _clause_to_dsl(
    clause: SignedClause, var_names: list[str], value_base: int = 0
) -> str:
    parts = []
    for lit in clause.literals:
        vs = ",".join(str(v + value_base) for v in sorted(lit.values))
        parts.append(f"[{vs}]:{var_names[lit.var]}")
    return " or ".join(parts)


def to_dsl(cc: CompiledConstraints) -> str:
    """Render a CompiledConstraints object as plain DSL text — re-parseable."""
    lines: list[str] = []
    for name, h in zip(cc.var_names, cc.var_domains):
        lines.append(f"var {name} {h}")
    if cc.value_base:
        lines.append("")
        lines.append(f"value-base {cc.value_base}")
    lines.append("")
    lines.append("ordering " + " ".join(cc.var_names[i] for i in cc.ordering))
    lines.append("")

    for pi_bar in cc.per_var_clauses:
        for clause in pi_bar:
            lines.append(_clause_to_dsl(clause, cc.var_names, cc.value_base))

    return "\n".join(lines).rstrip() + "\n"


def to_json(cc: CompiledConstraints) -> dict:
    return {
        "var_names": cc.var_names,
        "var_domains": cc.var_domains,
        "ordering": cc.ordering,
        "value_base": cc.value_base,
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
        value_base=int(d.get("value_base", 0)),
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
    p.add_argument(
        "--legacy-enum",
        action="store_true",
        help="Use the original per-clause combinations enumeration (kept for A/B).",
    )
    p.add_argument(
        "--ordering",
        choices=("file", "min-degree", "min-fill"),
        default="file",
        help="Elimination ordering source. 'file' uses cs.ordering from the input; "
        "'min-degree'/'min-fill' compute a heuristic ordering on the primal graph.",
    )
    p.add_argument(
        "--print-ordering",
        action="store_true",
        help="Print the elimination ordering (first-eliminated to last) and exit.",
    )
    args = p.parse_args()

    cs = parse_constraints(args.input.read_text())
    if args.ordering != "file":
        new_ordering = _heuristic_elimination_ordering(cs, args.ordering)
        if not args.quiet:
            print(
                f"# ordering source={args.ordering}: "
                + " ".join(cs.var_names[i] for i in reversed(new_ordering)),
                flush=True,
            )
        cs.ordering = new_ordering

    if args.print_ordering:
        names = [cs.var_names[i] for i in reversed(cs.ordering)]
        print(" ".join(names))
        return

    if args.output.suffix == ".txt":
        stream_to = args.output
    else:
        stream_to = args.output.with_suffix(".stream.txt")

    try:
        cc = compile_constraints(
            cs,
            verbose=not args.quiet,
            legacy_enum=args.legacy_enum,
            stream_to=stream_to,
        )
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
