from __future__ import annotations

import argparse
import io
import multiprocessing as mp
import os
import statistics
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "casper"))

from profile_compile_log import FIELDS  # noqa: E402
from resolution import _heuristic_elimination_ordering, compile_constraints  # noqa: E402

from generate_topology import generate_instance  # noqa: E402
from bench_topology_compile import induced_width  # noqa: E402

import re  # noqa: E402


def parse_steps(text: str) -> list[dict]:
    """Parse the compiler's verbose `step=` summary lines into dicts."""
    rows = []
    for line in text.splitlines():
        if not line.startswith("step="):
            continue
        row: dict = {}
        for f in FIELDS:
            m = re.search(rf"\b{f}=([0-9.]+)", line)
            if m:
                row[f] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
        rows.append(row)
    return rows


def compile_verbose(
    m: int, b: int, n_classes: int, seed: int, args, ordering: str = "min-fill"
) -> tuple[float, int, list[dict], str]:
    """Compile one instance with diagnostics on; return time, width, steps, log."""
    cs, _, _ = generate_instance(
        m, "exp", n_classes=n_classes, b=b, tau=args.tau,
        clauses_per_edge=args.clauses_per_edge, seed=seed,
    )
    if ordering != "file":
        cs.ordering = _heuristic_elimination_ordering(cs, ordering)
    width = induced_width(cs, cs.ordering)
    buf = io.StringIO()
    t0 = time.perf_counter()
    with redirect_stdout(buf):
        compile_constraints(cs, verbose=True)
    elapsed = time.perf_counter() - t0
    text = buf.getvalue()
    return elapsed, width, parse_steps(text), text


def _total(steps: list[dict], key: str) -> float:
    """Sum one per-step metric across the whole compile."""
    return sum(s.get(key, 0) for s in steps)


def report_steps(args) -> None:
    """Where compile time goes, and whether any cap or budget is binding.

    Distinguishes three candidate causes of the high-b plateau: a binding
    minimal-conflict arity cap (`J@cap` > 0), a binding expansion budget
    (`budget` > 0), and saturation of the derived clause set (raw resolvents
    growing while `kept` and `peak_pi` flatten).
    """
    print(
        f"SAT_MAX_EXPANSION={os.environ.get('SAT_MAX_EXPANSION', '<unset>')} "
        f"SAT_WORKERS={os.environ.get('SAT_WORKERS', '<unset>')} "
        f"(step cap = domain size K = {args.classes})"
    )
    print()
    print("config        median   IQR/med     raw    kept  sub_elim  peak_pi  "
          "maxJ  J@cap  expand%   sub%  budget")
    for m in args.m:
        for b in args.bands:
            times, rows, budget = [], [], 0
            for seed in range(args.trials):
                el, _w, steps, text = compile_verbose(
                    m, b, args.classes, seed, args
                )
                times.append(el)
                rows.append(steps)
                budget += text.count("expansion-budget")
            times_sorted = sorted(times)
            med = statistics.median(times)
            q1 = times_sorted[len(times_sorted) // 4]
            q3 = times_sorted[(3 * len(times_sorted)) // 4]
            raw = statistics.mean(_total(s, "n_resolvents_raw") for s in rows)
            kept = statistics.mean(_total(s, "n_resolvents_kept") for s in rows)
            sub = statistics.mean(_total(s, "sub_elim") for s in rows)
            peak = statistics.mean(
                max((x.get("pi_out_after_sub", 0) for x in s), default=0)
                for s in rows
            )
            maxj = max(
                max((x.get("max_J_size", 0) for x in s), default=0) for s in rows
            )
            jcap = sum(
                1 for s in rows for x in s
                if x.get("max_J_size", 0) >= args.classes
            )
            exp_t = statistics.mean(_total(s, "expand_elapsed_s") for s in rows)
            sub_t = statistics.mean(_total(s, "sub_elapsed_s") for s in rows)
            tot_t = max(statistics.mean(_total(s, "elapsed_s") for s in rows), 1e-9)
            print(
                f"m={m:<4} b={b}   {med:7.4f}s {(q3 - q1) / med:7.2f} "
                f"{raw:8.0f} {kept:7.0f} {sub:9.0f} {peak:8.0f} {maxj:5d} "
                f"{jcap:6d} {100 * exp_t / tot_t:7.1f} {100 * sub_t / tot_t:6.1f} "
                f"{budget:7d}"
            )


def report_ordering(args) -> None:
    """Check whether a particular elimination heuristic degenerates at high b."""
    print()
    print("ordering comparison (median over seeds)")
    for m in args.m:
        for b in args.bands:
            line = f"  m={m:<4} b={b}"
            for ordering in ("min-fill", "min-degree", "file"):
                res = [
                    compile_verbose(m, b, args.classes, s, args, ordering)
                    for s in range(args.trials)
                ]
                med = statistics.median(r[0] for r in res)
                widths = sorted({r[1] for r in res})
                line += f"   {ordering}: width={widths} {med:.4f}s"
            print(line)


def _ceiling_worker(m, b, k, seed, tau, cpe, q) -> None:
    """Compile one instance and post its raw/kept resolvent totals to `q`."""
    args = argparse.Namespace(tau=tau, clauses_per_edge=cpe)
    _el, _w, steps, _t = compile_verbose(m, b, k, seed, args)
    q.put((
        _total(steps, "n_resolvents_raw"),
        _total(steps, "n_resolvents_kept"),
        max((s.get("n_distinct_sets", 0) for s in steps), default=0),
    ))


def report_ceiling(args) -> None:
    """Test whether the high-b plateau is a domain-size ceiling.

    If the plateau comes from saturation of the space of distinct value-sets
    (bounded by K), then raising K should raise the level at which raw
    resolvents stop growing with b. Each compile runs in a subprocess so a
    blown-up (K, b) pair can be timed out instead of hanging the sweep.
    """
    print()
    print(f"resolvent ceiling vs domain size K (m={args.ceiling_m}, "
          f"timeout {args.ceiling_timeout}s)")
    print("   K    b    raw_resolvents   kept   max_distinct_sets")
    for k in args.ceiling_classes:
        for b in args.bands:
            q: mp.Queue = mp.Queue()
            p = mp.Process(
                target=_ceiling_worker,
                args=(args.ceiling_m, b, k, 0, args.tau, args.clauses_per_edge, q),
            )
            p.start()
            p.join(args.ceiling_timeout)
            if p.is_alive():
                p.terminate()
                p.join()
                print(f"  {k:2d}  {b:3d}    (timeout > {args.ceiling_timeout}s)")
                break
            raw, kept, dsets = q.get()
            print(f"  {k:2d}  {b:3d}    {raw:12.0f} {kept:6.0f} {dsets:15.0f}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Diagnose what drives (and caps) topology compile time."
    )
    p.add_argument("--m", type=int, nargs="+", default=[20, 200])
    p.add_argument("--bands", type=int, nargs="+", default=[3, 4, 5])
    p.add_argument("--classes", type=int, default=6)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--clauses-per-edge", type=int, default=2)
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--mode", nargs="+",
                   choices=("steps", "ordering", "ceiling"),
                   default=["steps", "ordering", "ceiling"])
    p.add_argument("--ceiling-classes", type=int, nargs="+", default=[4, 5, 6, 7])
    p.add_argument("--ceiling-m", type=int, default=20)
    p.add_argument("--ceiling-timeout", type=float, default=120.0)
    args = p.parse_args()

    if "steps" in args.mode:
        report_steps(args)
    if "ordering" in args.mode:
        report_ordering(args)
    if "ceiling" in args.mode:
        report_ceiling(args)


if __name__ == "__main__":
    main()
