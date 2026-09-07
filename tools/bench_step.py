import sys as _sys
from pathlib import Path as _Path

# Runnable entry point: put the shared compiler and this task's own directory on
# the path. Sibling task directories are deliberately not added -- they share
# module names (data, model, train, minedit).
_ROOT = _Path(__file__).resolve().parents[1]
for _d in (_ROOT / "casper", _ROOT / "examples"):
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))

#!/usr/bin/env python3
"""Replay a single variable-elimination step in isolation from a compiled file.

The compiled DSL stores clauses grouped per eliminated variable, and a
variable's group is exactly the `pi_bar` (clauses mentioning it) that fed its
elimination step. We recover that group and re-run `_grouped_eliminate_var`
on it alone, so the per-step resolvent enumeration can be A/B-profiled (e.g.
group-reordering on vs off via SAT_REORDER_GROUPS) without re-running the
whole multi-hour compile.

Ambient (pi_minus_bar) subsumption is dropped — it is not reconstructable from
the final artifact — so absolute counts differ from the full compile, but the
on/off comparison isolates the tautology-pruning effect, which is what changed.

Usage:
    python3 src/bench_step.py <compiled.txt> <var_name>
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from signed_clauses import parse_constraints  # noqa: E402
from resolution import (  # noqa: E402
    _grouped_eliminate_var,
    _REORDER_GROUPS_BY_SATURATION,
    _dedup_by_subsumption,
    _clause_to_dsl,
)


def extract_pi_bar(cs, var_name: str):
    """Return (var_index, pi_bar) for `var_name` from a compiled ConstraintSet.

    pi_bar = clauses mentioning the variable but none eliminated before it; in
    a finished compile that is precisely the variable's per-step input group.
    """
    var = cs.var_names.index(var_name)
    elim_order = list(reversed(cs.ordering))  # loop eliminates ordering[-1] first
    pos = elim_order.index(var)
    earlier = set(elim_order[:pos])  # variables eliminated before this one
    pi_bar = [
        c for c in cs.clauses
        if var in c.variables and not (c.variables & earlier)
    ]
    return var, pi_bar


def main() -> None:
    """Extract one variable's input group and run its elimination, timed."""
    # `--dump-raw` writes every generated raw resolvent (tagged kept/tautology/
    # duplicate/subsumed) to raw_resolvents_<var>.txt.
    argv = [a for a in sys.argv[1:] if a != "--dump-raw"]
    dump = "--dump-raw" in sys.argv
    compiled, var_name = argv[0], argv[1]
    # Optional positional arg: restrict to conflict patterns of exactly this
    # size (e.g. 3) so a single pattern class can be A/B-profiled quickly.
    only_size = int(argv[2]) if len(argv) > 2 else None
    cs = parse_constraints(Path(compiled).read_text())
    var, pi_bar = extract_pi_bar(cs, var_name)
    domains = list(cs.var_domains)

    print(
        f"# bench var={var_name} pi_bar={len(pi_bar)} "
        f"reorder={_REORDER_GROUPS_BY_SATURATION} cap={domains[var]} "
        f"only_pattern_size={only_size}",
        flush=True,
    )
    dump_raw: list | None = [] if dump else None
    t0 = time.time()
    kept, stats = _grouped_eliminate_var(
        pi_bar, [], var, domains,
        cap=domains[var], verbose=True, tag=f"bench var={var_name} ",
        only_pattern_size=only_size, dump_raw=dump_raw,
    )
    elapsed = time.time() - t0

    if dump_raw is not None:
        out_path = Path(f"raw_resolvents_{var_name}.txt")
        with out_path.open("w") as fh:
            fh.write(
                f"# {len(dump_raw)} raw resolvents for {var_name} "
                f"(tag = kept | tautology | duplicate | subsumed)\n"
            )
            for tag, clause in dump_raw:
                fh.write(
                    f"{tag}\t"
                    f"{_clause_to_dsl(clause, cs.var_names, cs.value_base)}\n"
                )
        print(f"# wrote {len(dump_raw)} raw resolvents -> {out_path}", flush=True)
    # Canonicalize to the subsumption-minimal set so the serial and parallel
    # paths report the same `kept` (parallel defers cross-chunk subsumption).
    kept = _dedup_by_subsumption(kept)
    print(
        f"# DONE var={var_name} reorder={_REORDER_GROUPS_BY_SATURATION} "
        f"kept={len(kept)} raw={stats['n_resolvents_raw']} "
        f"taut={stats['n_tautologies']} prefix_taut={stats['n_prefix_taut']} "
        f"pruned_leaves={stats['n_pruned_leaves']} "
        f"expand_s={stats['expand_elapsed_s']:.2f} total_s={elapsed:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
