from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "casper"))

from resolution import (  # noqa: E402
    Unsatisfiable,
    _heuristic_elimination_ordering,
    compile_constraints,
)
from signed_clauses import ConstraintSet  # noqa: E402

from generate_topology import class_means, generate_instance  # noqa: E402
from bench_topology_compile import _quantile, induced_width  # noqa: E402

STRATEGIES = ("structural", "high_conf", "low_conf", "random")


def confidence_score(X: np.ndarray, mus: np.ndarray) -> np.ndarray:
    """Negative distance to the nearest class centroid; higher = more confident."""
    dists = np.linalg.norm(X[:, None, :] - mus[None, :, :], axis=2)
    return -dists.min(axis=1)


def margin_score(X: np.ndarray, mus: np.ndarray) -> np.ndarray:
    """Distance to the second-nearest centroid minus the nearest one.

    Logged as a secondary confidence metric: a point equidistant from two
    centroids has a small margin even when its nearest-centroid distance is
    small, so the two metrics disagree exactly where confidence is ambiguous.
    """
    dists = np.sort(np.linalg.norm(X[:, None, :] - mus[None, :, :], axis=2), axis=1)
    return dists[:, 1] - dists[:, 0]


def elimination_sequence(
    cs: ConstraintSet,
    X: np.ndarray,
    mus: np.ndarray,
    strategy: str,
    rng: np.random.Generator,
    structural: str = "min-fill",
) -> list[int]:
    """Variable indices in the order they are eliminated (first eliminated first).

    `compile_constraints` eliminates `cs.ordering[-1]` first and `cs.ordering[0]`
    last, so callers must reverse this sequence before assigning it to
    `cs.ordering` — see `apply_ordering`.
    """
    if strategy == "structural":
        return list(reversed(_heuristic_elimination_ordering(cs, structural)))
    if strategy == "random":
        order = np.arange(len(X))
        rng.shuffle(order)
        return [int(v) for v in order]
    conf = confidence_score(X, mus)
    if strategy == "high_conf":
        return [int(v) for v in np.argsort(-conf)]
    if strategy == "low_conf":
        return [int(v) for v in np.argsort(conf)]
    raise ValueError(f"unknown strategy {strategy!r}")


def apply_ordering(cs: ConstraintSet, elim_seq: list[int]) -> None:
    """Install an elimination sequence on `cs`, honouring the reversed convention."""
    cs.ordering = list(reversed(elim_seq))


def _compile_worker(cs: ConstraintSet, queue: mp.Queue) -> None:
    """Compile in a child process and post the measured metrics back."""
    try:
        t0 = time.perf_counter()
        cc = compile_constraints(cs, verbose=False)
        elapsed = time.perf_counter() - t0
        originals = set(cs.clauses)
        queue.put({
            "compile_s": elapsed,
            "clauses_out": sum(len(b) for b in cc.per_var_clauses),
            "resolvents": sum(
                1 for b in cc.per_var_clauses for c in b if c not in originals
            ),
            "max_bucket": max(len(b) for b in cc.per_var_clauses),
        })
    except Unsatisfiable:
        queue.put({"unsat": True})


def compile_with_timeout(cs: ConstraintSet, timeout: float | None) -> dict | None:
    """Compile `cs`, returning None if it exceeds `timeout` seconds.

    A `timeout` of None waits indefinitely, so every reported number is a true
    measurement rather than a lower bound. The compile runs in a forked child so
    a blown-up ordering can be abandoned when a cap is set; timing is taken
    inside the child, so the fork itself is not charged to the measurement.
    """
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_compile_worker, args=(cs, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return None
    return queue.get() if not queue.empty() else None


def run_one(m: int, graph: str, b: int, strategy: str, args, seed: int) -> dict | None:
    """Compile one instance under one elimination strategy; None if censored."""
    cs, X, _labels = generate_instance(
        m, graph, n_classes=args.classes, b=b, tau=args.tau,
        clauses_per_edge=args.clauses_per_edge, theta=args.theta,
        radius=args.radius, lam=args.lam, eps=args.eps,
        noise_frac=args.noise_frac, tree_shape=args.tree_shape, seed=seed,
    )
    mus = class_means(args.classes, args.radius)
    rng = np.random.default_rng(seed)
    apply_ordering(
        cs, elimination_sequence(cs, X, mus, strategy, rng, args.structural)
    )
    width = induced_width(cs, cs.ordering)

    result = compile_with_timeout(cs, args.per_compile_timeout)
    if result is None or result.get("unsat"):
        return None
    return {
        "m": m, "graph": graph, "band": b if graph == "exp" else 0,
        "ordering": strategy, "seed": seed, "induced_width": width,
        "clauses_in": len(cs.clauses),
        "mean_margin": float(margin_score(X, mus).mean()),
        **result,
    }


def aggregate(trials: list[dict], censored: int) -> dict:
    """Collapse per-seed trials of one (config, ordering) cell into one row."""
    times = sorted(t["compile_s"] for t in trials)
    head = trials[0]
    return {
        "m": head["m"], "graph": head["graph"], "band": head["band"],
        "ordering": head["ordering"],
        "confidence_metric": "neg_nearest_centroid_distance",
        "trials": len(trials), "censored": censored,
        "induced_width": round(
            statistics.mean(t["induced_width"] for t in trials), 1
        ),
        "clauses_in": head["clauses_in"],
        "clauses_out": round(statistics.mean(t["clauses_out"] for t in trials)),
        "resolvents": round(statistics.mean(t["resolvents"] for t in trials)),
        "max_bucket": max(t["max_bucket"] for t in trials),
        "mean_margin": round(
            statistics.mean(t["mean_margin"] for t in trials), 3
        ),
        "compile_s": round(statistics.mean(times), 4),
        "compile_std": round(
            statistics.stdev(times) if len(times) > 1 else 0.0, 4
        ),
        "compile_med": round(statistics.median(times), 4),
        "compile_p25": round(_quantile(times, 0.25), 4),
        "compile_p75": round(_quantile(times, 0.75), 4),
        "compile_min": round(times[0], 4),
        "compile_max": round(times[-1], 4),
    }


def write_csv(rows: list[dict], out: Path) -> None:
    """Write one row per (m, graph, band, ordering) cell."""
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote CSV -> {out}")


def append_csv(row: dict, out: Path) -> None:
    """Append one finished cell to the CSV immediately.

    Written as each cell completes so an uncapped sweep that is interrupted (or
    that stalls on a pathological ordering) still leaves every result it has
    already produced on disk.
    """
    exists = out.exists()
    with out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def write_markdown(rows: list[dict], out: Path, args) -> None:
    """Write the comparison as a Markdown table, ordering-major within config."""
    lines = [
        "# Elimination ordering by confidence vs. structural heuristics",
        "",
        f"The constraint graph is held fixed within each row group; the only "
        f"independent variable is the elimination order. K={args.classes}, "
        f"tau={args.tau}, lambda={args.lam}, epsilon={args.eps}. Confidence is "
        f"the negative distance to the nearest class centroid; `structural` is "
        f"CASPER's `{args.structural}` heuristic on the primal graph.",
        "",
        "`width` is the induced width under the ordering actually used and is "
        "the mechanism to watch: a confidence ranking ignores the graph, so it "
        "controls compile cost only insofar as it happens to correlate with a "
        "good structural order.",
        "",
        "| topology | b | m | ordering | width | resolvents | max bucket | "
        "compile (median) | IQR | vs structural | censored |",
        "|:--|---:|---:|:--|---:|---:|---:|---:|---:|---:|---:|",
    ]
    base = {
        (r["graph"], r["band"], r["m"]): r["compile_med"]
        for r in rows if r["ordering"] == "structural"
    }
    for r in rows:
        band = str(r["band"]) if r["graph"] == "exp" else "-"
        ref = base.get((r["graph"], r["band"], r["m"]))
        rel = f"{r['compile_med'] / ref:.2f}x" if ref else "-"
        lines.append(
            f"| `{r['graph']}` | {band} | {r['m']} | `{r['ordering']}` | "
            f"{r['induced_width']} | {r['resolvents']:,} | {r['max_bucket']:,} | "
            f"{r['compile_med']:.4f}s | "
            f"{r['compile_p25']:.4f}-{r['compile_p75']:.4f} | {rel} | "
            f"{r['censored']} |"
        )
    lines += [
        "",
        "`vs structural` is the median compile time relative to the structural "
        "baseline on the identical constraint graph; >1 means the ordering is "
        "worse than the heuristic CASPER already uses. `censored` counts seeds "
        "abandoned at the per-compile timeout "
        f"({args.per_compile_timeout}s) and excluded from the statistics, so a "
        "cell with censored > 0 understates that ordering's true cost.",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote Markdown -> {out}")


def make_png(rows: list[dict], out: Path, args) -> None:
    """Plot compile time and induced width per ordering strategy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "structural": "#2980b9", "high_conf": "#c0392b",
        "low_conf": "#e67e22", "random": "#7f8c8d",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    exp_rows = [r for r in rows if r["graph"] == "exp"]
    m_star = max((r["m"] for r in exp_rows), default=0)

    ax = axes[0]
    for strat in STRATEGIES:
        sub = sorted(
            (r for r in exp_rows if r["ordering"] == strat and r["m"] == m_star),
            key=lambda r: r["band"],
        )
        if sub:
            ax.plot([r["band"] for r in sub], [r["compile_med"] for r in sub],
                    "o-", color=colors[strat], lw=2, ms=5, label=strat)
    ax.set_yscale("log")
    ax.set_xlabel("bandwidth b")
    ax.set_ylabel("compile wall time (s)")
    ax.set_title(f"`exp` compile time by ordering (m={m_star})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for strat in STRATEGIES:
        sub = sorted(
            (r for r in exp_rows if r["ordering"] == strat and r["m"] == m_star),
            key=lambda r: r["band"],
        )
        if sub:
            ax.plot([r["band"] for r in sub], [r["induced_width"] for r in sub],
                    "o-", color=colors[strat], lw=2, ms=5, label=strat)
    ax.set_xlabel("bandwidth b")
    ax.set_ylabel("induced width")
    ax.set_title(f"Induced width by ordering (m={m_star})")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    poly = [r for r in rows if r["graph"] == "poly"]
    for strat in STRATEGIES:
        sub = sorted(
            (r for r in poly if r["ordering"] == strat), key=lambda r: r["m"]
        )
        if sub:
            ax.plot([r["m"] for r in sub], [r["compile_med"] for r in sub],
                    "o-", color=colors[strat], lw=2, ms=5, label=strat)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of variables m")
    ax.set_ylabel("compile wall time (s)")
    ax.set_title("`poly` (tree) compile time by ordering")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Confidence-ranked vs. structural elimination ordering "
        f"(K={args.classes}, tau={args.tau}, graph held fixed)", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Wrote plot -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare confidence-ranked elimination orderings against "
                    "CASPER's structural heuristic on a fixed constraint graph."
    )
    p.add_argument("--m", type=int, nargs="+", default=[20, 50, 100, 200])
    p.add_argument("--graphs", nargs="+", choices=("exp", "poly"),
                   default=["exp", "poly"])
    p.add_argument("--bands", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--orderings", nargs="+", choices=STRATEGIES,
                   default=list(STRATEGIES))
    p.add_argument("--structural", choices=("min-fill", "min-degree"),
                   default="min-fill")
    p.add_argument("--classes", type=int, default=6)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--clauses-per-edge", type=int, default=2)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--heavy-trials", type=int, default=15)
    p.add_argument("--heavy-from-band", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--theta", type=float, default=1.0)
    p.add_argument("--radius", type=float, default=10.0)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--noise-frac", type=float, default=0.5)
    p.add_argument("--tree-shape", choices=("random", "star", "path"),
                   default="random")
    p.add_argument("--per-compile-timeout", type=float, default=60.0,
                   help="Abandon a single compile after this many seconds; "
                        "0 or less means no cap (every number is then a true "
                        "measurement, but a single compile can run for hours).")
    p.add_argument("--prefix", type=Path, default=Path("compile_ordering_confidence"))
    p.add_argument("--no-png", action="store_true")
    args = p.parse_args()
    if args.per_compile_timeout <= 0:
        args.per_compile_timeout = None

    configs: list[tuple[str, int]] = []
    if "poly" in args.graphs:
        configs.append(("poly", 0))
    if "exp" in args.graphs:
        configs += [("exp", b) for b in args.bands]

    live_csv = args.prefix.with_suffix(".csv")
    if live_csv.exists():
        live_csv.unlink()

    rows: list[dict] = []
    for m in sorted(args.m):
        for graph, b in configs:
            if graph == "exp" and b >= m:
                continue
            for strategy in args.orderings:
                n_trials = (
                    args.heavy_trials
                    if graph == "exp" and b >= args.heavy_from_band
                    else args.trials
                )
                trials, censored = [], 0
                for t in range(n_trials):
                    got = run_one(m, graph, b, strategy, args, args.seed + t)
                    if got is None:
                        censored += 1
                    else:
                        trials.append(got)
                if not trials:
                    print(
                        f"{graph:<4} b={b} m={m:<4} {strategy:<11} "
                        f"all {censored} seeds censored", flush=True,
                    )
                    continue
                row = aggregate(trials, censored)
                rows.append(row)
                append_csv(row, live_csv)
                print(
                    f"{graph:<4} b={b} m={m:<4} {strategy:<11} "
                    f"width={row['induced_width']:<6} n={row['trials']:<3} "
                    f"cens={row['censored']:<2} res={row['resolvents']:<7} "
                    f"median={row['compile_med']:.4f}s "
                    f"IQR=[{row['compile_p25']:.4f},{row['compile_p75']:.4f}]",
                    flush=True,
                )

    if not rows:
        raise SystemExit("no configs completed")

    write_csv(rows, args.prefix.with_suffix(".csv"))
    write_markdown(rows, args.prefix.with_suffix(".md"), args)
    if not args.no_png:
        make_png(rows, args.prefix.with_suffix(".png"), args)


if __name__ == "__main__":
    main()
