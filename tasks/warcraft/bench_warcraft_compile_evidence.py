from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "casper"))
sys.path.insert(0, str(ROOT / "examples"))

from constraints import Constraints  # noqa: E402
from signed_clauses import parse_constraints  # noqa: E402
from warcraft_degree_constraints import (  # noqa: E402
    build_ordering,
    degree_clauses,
    induced_width,
    node_incident,
    to_dsl_text,
)


def dp_count_degree_legal(N):
    """Independent exact count of degree-legal edge sets, by transfer matrix.

    Rows are processed in turn; the state is which vertical edges enter the
    current row. This shares no code with the clause machinery, so it is a
    genuine second opinion rather than a restatement of the same logic.
    """
    src, sink = (0, 0), (N - 1, N - 1)
    states = {0: 1}  # bitmask of vertical edges entering row 0 -> count
    for i in range(N):
        nxt = {}
        last_row = (i == N - 1)
        d_range = [0] if last_row else range(1 << N)
        for v, cnt in states.items():
            for h in range(1 << (N - 1)):
                for d in d_range:
                    ok = True
                    for j in range(N):
                        deg = ((v >> j) & 1) + ((d >> j) & 1)
                        if j > 0:
                            deg += (h >> (j - 1)) & 1
                        if j < N - 1:
                            deg += (h >> j) & 1
                        node = (i, j)
                        if node in (src, sink):
                            if deg != 1:
                                ok = False
                                break
                        elif deg not in (0, 2):
                            ok = False
                            break
                    if ok:
                        nxt[d] = nxt.get(d, 0) + cnt
        states = nxt
    return states.get(0, 0)


def brute_force_count(N):
    """Exhaustive 2^E sweep. Only tractable for very small N."""
    inc = node_incident(N)
    n_edges = 2 * N * (N - 1)
    nodes = sorted(inc)
    A = np.zeros((len(nodes), n_edges), dtype=np.int8)
    for r, node in enumerate(nodes):
        for e in inc[node]:
            A[r, e] = 1
    endpoint = np.array([nd in ((0, 0), (N - 1, N - 1)) for nd in nodes])
    total = 1 << n_edges
    chunk = min(total, 1 << 20)
    bits = np.arange(n_edges, dtype=np.int64)
    count = 0
    for start in range(0, total, chunk):
        idx = np.arange(start, min(start + chunk, total), dtype=np.int64)
        X = ((idx[:, None] >> bits[None, :]) & 1).astype(np.int8)
        deg = X @ A.T
        ok = (deg[:, endpoint] == 1).all(axis=1)
        other = deg[:, ~endpoint]
        ok &= ((other == 0) | (other == 2)).all(axis=1)
        count += int(ok.sum())
    return count


def independent_degree_check(model, N):
    """Verify one enumerated model is degree-legal, without touching clauses."""
    inc = node_incident(N)
    for node, edges in inc.items():
        deg = sum(model[e] for e in edges)
        if node in ((0, 0), (N - 1, N - 1)):
            if deg != 1:
                return False
        elif deg not in (0, 2):
            return False
    return True


def walk(raw, compiled, n_edges, N, cap=None, verify_degrees=True):
    """Enumerate the raw system, comparing feasible sets against the compiled one.

    Returns model count, dead ends, empty feasible sets, the number of prefixes
    where the two disagree, and the set of resolvents that ever bind.
    """
    ordering = list(raw.ordering)
    n = len(ordering)
    stats = {"models": 0, "dead": 0, "empty": 0, "prefix_diffs": 0,
             "unsound_models": 0, "capped": False}
    binding = set()

    raw_clause_set = None
    resolvents_by_step = None
    if compiled is not None:
        raw_clause_set = {
            frozenset((lit.var, tuple(sorted(lit.values))) for lit in c.literals)
            for pi in raw._impl.compiled.per_var_clauses for c in pi
        }
        resolvents_by_step = []
        for pi in compiled._impl.compiled.per_var_clauses:
            here = []
            for c in pi:
                key = frozenset(
                    (lit.var, tuple(sorted(lit.values))) for lit in c.literals)
                if key not in raw_clause_set:
                    here.append((key, c))
            resolvents_by_step.append(here)

    def rec(step, a):
        if stats["capped"]:
            return 0
        if step == n:
            stats["models"] += 1
            if verify_degrees:
                model = [a[v] for v in range(n_edges)]
                if not independent_degree_check(model, N):
                    stats["unsound_models"] += 1
            if cap and stats["models"] >= cap:
                stats["capped"] = True
            return 1

        fs_raw = raw.feasible_set(step, a)
        if not fs_raw:
            stats["empty"] += 1

        if compiled is not None:
            fs_cmp = compiled.feasible_set(step, a)
            if fs_raw != fs_cmp:
                stats["prefix_diffs"] += 1
            for key, clause in resolvents_by_step[step]:
                var = ordering[step]
                satisfied = False
                for lit in clause.literals:
                    if lit.var == var:
                        continue
                    if lit.var in a and a[lit.var] in lit.values:
                        satisfied = True
                        break
                if satisfied:
                    continue
                s_j = clause.get(var) or frozenset()
                if (fs_raw & s_j) != fs_raw:
                    binding.add(key)

        found = 0
        for val in sorted(fs_raw):
            a[ordering[step]] = val
            got = rec(step + 1, a)
            if got == 0 and not stats["capped"]:
                stats["dead"] += 1
            found += got
            del a[ordering[step]]
        return found

    rec(0, {})
    return stats, binding


def walk_raw_only(raw, n_edges, N, degree_check_every=100):
    """Full enumeration of the raw system, without the compiled comparison.

    Used where the per-prefix comparison would be quadratic in the resolvent
    count; the model total is still cross-checked against the DP oracle.
    """
    ordering = list(raw.ordering)
    n = len(ordering)
    stats = {"models": 0, "dead": 0, "empty": 0, "unsound_models": 0,
             "degree_checked": 0}

    def rec(step, a):
        if step == n:
            stats["models"] += 1
            if stats["models"] % degree_check_every == 0:
                stats["degree_checked"] += 1
                if not independent_degree_check([a[v] for v in range(n_edges)], N):
                    stats["unsound_models"] += 1
            return 1
        fs = raw.feasible_set(step, a)
        if not fs:
            stats["empty"] += 1
        found = 0
        for val in sorted(fs):
            a[ordering[step]] = val
            got = rec(step + 1, a)
            if got == 0:
                stats["dead"] += 1
            found += got
            del a[ordering[step]]
        return found

    rec(0, {})
    return stats


def sample_compare(raw, compiled, n_edges, N, k, seed=0):
    """Compare raw vs compiled feasible sets along k random complete walks."""
    import random

    rng = random.Random(seed)
    ordering = list(raw.ordering)
    n = len(ordering)

    raw_keys = {
        frozenset((lit.var, tuple(sorted(lit.values))) for lit in c.literals)
        for pi in raw._impl.compiled.per_var_clauses for c in pi
    }
    resolvents_by_step = []
    for pi in compiled._impl.compiled.per_var_clauses:
        here = []
        for c in pi:
            key = frozenset(
                (lit.var, tuple(sorted(lit.values))) for lit in c.literals)
            if key not in raw_keys:
                here.append((key, c))
        resolvents_by_step.append(here)

    binding = set()
    prefixes = diffs = unsound = 0
    for _ in range(k):
        a = {}
        for step in range(n):
            var = ordering[step]
            fs_raw = raw.feasible_set(step, a)
            fs_cmp = compiled.feasible_set(step, a)
            prefixes += 1
            if fs_raw != fs_cmp:
                diffs += 1
            for key, clause in resolvents_by_step[step]:
                satisfied = False
                for lit in clause.literals:
                    if lit.var == var:
                        continue
                    if lit.var in a and a[lit.var] in lit.values:
                        satisfied = True
                        break
                if satisfied:
                    continue
                if (fs_raw & (clause.get(var) or frozenset())) != fs_raw:
                    binding.add(key)
            if not fs_raw:
                break
            a[var] = rng.choice(sorted(fs_raw))
        else:
            if not independent_degree_check([a[v] for v in range(n_edges)], N):
                unsound += 1
    return {"prefixes": prefixes, "diffs": diffs, "unsound": unsound}, binding


def run_compile(N, src_txt, out_json, timeout_s):
    """Attempt the exact compile; return (total, resolvents, seconds) or None."""
    env_prefix = ["env", "SAT_ORBIT_ENUM=1", "SAT_ORBIT_EXPAND=0"]
    t0 = time.perf_counter()
    try:
        r = subprocess.run(
            env_prefix + [sys.executable, str(ROOT / "casper" / "resolution.py"),
                          str(src_txt), str(out_json), "--quiet"],
            capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None, None, timeout_s, "timeout"
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        return None, None, dt, f"exit {r.returncode}"
    total = resolvents = None
    for line in r.stdout.splitlines():
        if line.startswith("Compiled "):
            parts = line.replace("(", " ").replace(")", " ").split()
            total = int(parts[1])
            resolvents = int(parts[parts.index("+") + 1])
    return total, resolvents, dt, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--compile-timeout", type=int, default=900)
    ap.add_argument("--brute-max-n", type=int, default=4)
    ap.add_argument("--walk-cap", type=int, default=20_000_000)
    ap.add_argument("--sampled-grids", type=int, nargs="*", default=[6],
                    help="Grids whose raw-vs-compiled comparison is sampled "
                         "rather than exhaustive.")
    ap.add_argument("--sample-walks", type=int, default=20000)
    ap.add_argument("--outdir", type=Path, default=ROOT / "constraints" / "warcraft" / "warcraft_artifacts")
    ap.add_argument("--out-json", type=Path,
                    default=ROOT / "logs" / "warcraft_compile_evidence.json")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    for N in args.grids:
        n_edges = 2 * N * (N - 1)
        raw_n = len(degree_clauses(N))
        ordering = build_ordering(N, "geometric")
        width, _ = induced_width(N, ordering)
        text, _ = to_dsl_text(N, ordering)
        src_txt = args.outdir / f"wc_deg_{N}_geometric.txt"
        src_txt.write_text(text)
        print(f"\n=== N={N}  edges={n_edges}  raw_clauses={raw_n}  width={width} ===",
              flush=True)

        # Independent oracle.
        t0 = time.perf_counter()
        dp = dp_count_degree_legal(N)
        dp_s = time.perf_counter() - t0
        print(f"  DP count            : {dp}  [{dp_s:.2f}s]", flush=True)

        bf = None
        bf_s = None
        if N <= args.brute_max_n:
            t0 = time.perf_counter()
            bf = brute_force_count(N)
            bf_s = time.perf_counter() - t0
            print(f"  brute force 2^{n_edges}   : {bf}  [{bf_s:.2f}s]", flush=True)

        # Exact compile.
        out_json = args.outdir / f"wc_deg_{N}_geometric_compiled.json"
        total, resolvents, comp_s, status = run_compile(
            N, src_txt, out_json, args.compile_timeout)
        print(f"  compile             : status={status} total={total} "
              f"resolvents={resolvents} [{comp_s:.1f}s]", flush=True)

        raw_cons = Constraints.from_compiled_dsl(src_txt)
        comp_cons = Constraints.from_file(out_json) if status == "ok" else None

        sampled = N in args.sampled_grids
        sample_stats = None
        if sampled:
            # Full enumeration for the model/dead-end totals, sampled walks for
            # the raw-vs-compiled comparison (which is O(models x resolvents)).
            t0 = time.perf_counter()
            raw_stats = walk_raw_only(raw_cons, n_edges, N)
            walk_s = time.perf_counter() - t0
            stats = {"models": raw_stats["models"], "dead": raw_stats["dead"],
                     "empty": raw_stats["empty"],
                     "unsound_models": raw_stats["unsound_models"],
                     "prefix_diffs": None, "capped": False}
            print(f"  walk (full, raw)    : models={stats['models']} "
                  f"dead={stats['dead']} empty={stats['empty']} "
                  f"unsound={stats['unsound_models']}/"
                  f"{raw_stats['degree_checked']} sampled-degree-checked "
                  f"[{walk_s:.1f}s]", flush=True)
            binding = set()
            if comp_cons is not None:
                t0 = time.perf_counter()
                sample_stats, binding = sample_compare(
                    raw_cons, comp_cons, n_edges, N, args.sample_walks)
                print(f"  compare (sampled)   : walks={args.sample_walks} "
                      f"prefixes={sample_stats['prefixes']} "
                      f"diffs={sample_stats['diffs']} "
                      f"binding_resolvents={len(binding)} "
                      f"unsound={sample_stats['unsound']} "
                      f"[{time.perf_counter() - t0:.1f}s]", flush=True)
        else:
            t0 = time.perf_counter()
            stats, binding = walk(raw_cons, comp_cons, n_edges, N, cap=args.walk_cap)
            walk_s = time.perf_counter() - t0
            print(f"  walk                : models={stats['models']} "
                  f"dead={stats['dead']} empty={stats['empty']} "
                  f"prefix_diffs={stats['prefix_diffs']} "
                  f"binding_resolvents={len(binding)} "
                  f"unsound={stats['unsound_models']} capped={stats['capped']} "
                  f"[{walk_s:.1f}s]", flush=True)

        rows.append({
            "sampled_comparison": sample_stats,
            "N": N, "edges": n_edges, "raw_clauses": raw_n, "width": width,
            "compile_status": status, "compiled_total": total,
            "resolvents": resolvents, "compile_s": comp_s,
            "binding_resolvents": len(binding),
            "prefix_diffs": stats["prefix_diffs"],
            "walk_models": stats["models"], "walk_capped": stats["capped"],
            "walk_s": walk_s, "dead_ends": stats["dead"],
            "empty_feasible": stats["empty"],
            "unsound_models": stats["unsound_models"],
            "dp_count": dp, "dp_s": dp_s,
            "brute_force_count": bf, "brute_force_s": bf_s,
        })

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    main()
