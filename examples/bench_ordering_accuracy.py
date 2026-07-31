from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from constraints import Constraints  # noqa: E402
from resolution import _heuristic_elimination_ordering, compile_constraints  # noqa: E402
from signed_clauses import ConstraintSet  # noqa: E402

from generate_topology import class_means, generate_instance  # noqa: E402
from check_constraint_bite import class_scores, infer_walk  # noqa: E402

STRATEGIES = ("min_fill", "high_conf", "low_conf", "random")


def assignment_order(
    cs: ConstraintSet,
    conf: np.ndarray,
    strategy: str,
    rng: np.random.Generator,
) -> list[int]:
    """Order in which variables are assigned by the INFER walk.

    The walk consumes `cs.ordering` front-to-back, so index 0 is assigned
    first: `high_conf` puts the most confident variable at the front. (The
    compile eliminates the same list back-to-front, which is why this is the
    reverse of the convention used in the compile-time ordering study.)
    """
    if strategy == "min_fill":
        return _heuristic_elimination_ordering(cs, "min-fill")
    if strategy == "random":
        order = np.arange(len(conf))
        rng.shuffle(order)
        return [int(v) for v in order]
    if strategy == "high_conf":
        return [int(v) for v in np.argsort(-conf)]
    if strategy == "low_conf":
        return [int(v) for v in np.argsort(conf)]
    raise ValueError(f"unknown strategy {strategy!r}")


def run_instance(args, m: int, graph: str, b: int, lam: float, seed: int) -> dict:
    """Score every ordering strategy on one shared instance."""
    cs, X, labels = generate_instance(
        m, graph, n_classes=args.classes, b=b, tau=args.tau,
        clauses_per_edge=args.clauses_per_edge, theta=args.theta,
        radius=args.radius, lam=lam, eps=args.eps,
        noise_frac=args.noise_frac, noise_dist=args.noise_dist, seed=seed,
    )
    mus = class_means(args.classes, args.radius)
    scores = class_scores(X, mus, args.temp)
    conf = scores.max(axis=1)
    raw_pred = scores.argmax(axis=1)
    base_acc = float((raw_pred == labels).mean())

    out = {"base_acc": base_acc}
    rng = np.random.default_rng(seed)
    for strategy in STRATEGIES:
        cs.ordering = assignment_order(cs, conf, strategy, rng)
        cons = Constraints.from_compiled(compile_constraints(cs, verbose=False))
        assignment, bites, total_feasible = infer_walk(cons, scores)
        pred = np.array([assignment[i] for i in range(m)])
        changed = pred != raw_pred
        out[strategy] = {
            "acc": float((pred == labels).mean()),
            "fixed": float(
                ((pred == labels) & (raw_pred != labels) & changed).sum()
            ) / m,
            "broke": float(
                ((pred != labels) & (raw_pred == labels) & changed).sum()
            ) / m,
            "bite": bites / m,
            "mean_feasible": total_feasible / m,
        }
    return out


def _sd(vals: list[float]) -> float:
    """Sample standard deviation, 0 for a single observation."""
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def summarise(per_seed: list[dict], m: int, graph: str, b: int,
              lam: float) -> list[dict]:
    """Aggregate the per-seed runs into one row per ordering strategy."""
    base = [r["base_acc"] for r in per_seed]
    rows = []
    for strategy in STRATEGIES:
        accs = [r[strategy]["acc"] for r in per_seed]
        deltas = [r[strategy]["acc"] - r["base_acc"] for r in per_seed]
        rows.append({
            "graph": graph, "band": b, "m": m, "lam": lam,
            "ordering": strategy, "seeds": len(per_seed),
            "base_acc": statistics.mean(base),
            "acc": statistics.mean(accs), "acc_sd": _sd(accs),
            "delta": statistics.mean(deltas), "delta_sd": _sd(deltas),
            "fixed": statistics.mean(r[strategy]["fixed"] for r in per_seed),
            "broke": statistics.mean(r[strategy]["broke"] for r in per_seed),
            "bite": statistics.mean(r[strategy]["bite"] for r in per_seed),
            "mean_feasible": statistics.mean(
                r[strategy]["mean_feasible"] for r in per_seed
            ),
        })
    return rows


def paired_rows(per_seed: list[dict], m: int, graph: str, b: int,
                lam: float) -> list[dict]:
    """Paired high-vs-low / high-vs-random contrasts on identical instances."""
    out = []
    for a, c in (("high_conf", "low_conf"), ("high_conf", "random"),
                 ("high_conf", "min_fill")):
        diffs = [r[a]["acc"] - r[c]["acc"] for r in per_seed]
        mean, sd = statistics.mean(diffs), _sd(diffs)
        n = len(diffs)
        t = mean / (sd / math.sqrt(n)) if sd > 0 else float("inf") if mean else 0.0
        out.append({
            "graph": graph, "band": b, "m": m, "lam": lam,
            "contrast": f"{a} - {c}", "seeds": n,
            "mean_diff": mean, "sd": sd, "t": t,
            "wins": sum(1 for d in diffs if d > 0),
            "ties": sum(1 for d in diffs if d == 0),
            "losses": sum(1 for d in diffs if d < 0),
        })
    return out


def aligned_table(headers: list[str], body: list[list[str]],
                  aligns: list[str]) -> list[str]:
    """Render a padded Markdown table."""
    widths = [
        max(len(h), max((len(r[i]) for r in body), default=0))
        for i, h in enumerate(headers)
    ]

    def row(cells: list[str]) -> str:
        return "| " + " | ".join(
            c.ljust(w) if a == "left" else c.rjust(w)
            for c, w, a in zip(cells, widths, aligns)
        ) + " |"

    sep = "|" + "|".join(
        ":" + "-" * (w + 1) if a == "left" else "-" * (w + 1) + ":"
        for w, a in zip(widths, aligns)
    ) + "|"
    return [row(headers), sep] + [row(c) for c in body]


def write_markdown(rows: list[dict], pairs: list[dict], out: Path, args) -> None:
    """Write the accuracy comparison and the paired contrasts."""
    lines = [
        "# Does assigning high-confidence variables first improve accuracy?",
        "",
        f"Constraint graph, data and classifier scores are identical across the "
        f"four orderings within each row group; only the INFER assignment order "
        f"differs. K={args.classes}, tau={args.tau}, "
        f"{args.clauses_per_edge} clauses/edge, epsilon={args.eps}, "
        f"noise_frac={args.noise_frac}, noise_dist={args.noise_dist}, "
        f"temp={args.temp}. Accuracy is the "
        "fraction of variables assigned their true class.",
        "",
        "`base %` is the unconstrained argmax of the classifier scores — the "
        "accuracy with no constraint layer at all, identical for every "
        "ordering. `Δ %` is each ordering's accuracy relative to it. "
        "`seeds` is the number of "
        "independently generated instances; all four orderings see the same "
        "instance within a seed, which is what makes the contrasts below "
        f"paired. `min_fill` is CASPER's structural heuristic on the primal "
        "graph — the ordering the compiler would pick on its own.",
        "",
    ]
    body = []
    for r in rows:
        band = str(r["band"]) if r["graph"] == "exp" else "-"
        body.append([
            "E" if r["graph"] == "exp" else "P", band, str(r["m"]),
            f"{r['lam']:.2f}", r["ordering"], str(r["seeds"]),
            f"{r['base_acc'] * 100:.2f}", f"{r['delta'] * 100:+.2f}",
            f"{r['acc'] * 100:.2f} ± {r['acc_sd'] * 100:.2f}",
        ])
    lines += aligned_table(
        ["topology", "b", "n", "λ", "ordering", "seeds", "base %", "Δ %",
         "acc % (mean ± sd)"],
        body, ["left", "right", "right", "right", "left"] + ["right"] * 4,
    )

    lines += [
        "",
        "## Paired contrasts",
        "",
        "Each contrast is computed per instance and then averaged, so the two "
        "orderings being compared always saw the identical graph, data and "
        "scores. `t` is the paired t-statistic; `w/t/l` counts instances where "
        "the first ordering won, tied or lost.",
        "",
    ]
    pbody = []
    for p in pairs:
        band = str(p["band"]) if p["graph"] == "exp" else "-"
        pbody.append([
            "E" if p["graph"] == "exp" else "P", band, str(p["m"]),
            f"{p['lam']:.2f}", p["contrast"], str(p["seeds"]),
            f"{p['wins']}/{p['ties']}/{p['losses']}",
            "inf" if math.isinf(p["t"]) else f"{p['t']:+.2f}",
            f"{p['mean_diff'] * 100:+.2f} ± {p['sd'] * 100:.2f}",
        ])
    lines += aligned_table(
        ["topology", "b", "n", "λ", "contrast", "seeds", "w/t/l", "t",
         "Δacc % (mean ± sd)"],
        pbody, ["left", "right", "right", "right", "left"] + ["right"] * 4,
    )
    lines.append("")
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote Markdown -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare INFER assignment orderings on task accuracy."
    )
    p.add_argument("--m", type=int, nargs="+", default=[50, 100])
    p.add_argument("--graphs", nargs="+", choices=("exp", "poly"),
                   default=["poly", "exp"])
    p.add_argument("--bands", type=int, nargs="+", default=[2])
    p.add_argument("--lams", type=float, nargs="+", default=[0.9, 1.0])
    p.add_argument("--classes", type=int, default=6)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--clauses-per-edge", type=int, default=2)
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--noise-frac", type=float, default=0.5)
    p.add_argument("--noise-dist", choices=("gaussian", "uniform"),
                   default="gaussian")
    p.add_argument("--theta", type=float, default=1.0)
    p.add_argument("--radius", type=float, default=10.0)
    p.add_argument("--temp", type=float, default=10.0)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--prefix", type=Path, default=Path("ordering_accuracy"))
    args = p.parse_args()

    configs: list[tuple[str, int]] = []
    if "poly" in args.graphs:
        configs.append(("poly", 0))
    if "exp" in args.graphs:
        configs += [("exp", b) for b in args.bands]

    rows: list[dict] = []
    pairs: list[dict] = []
    for lam in args.lams:
        for m in sorted(args.m):
            for graph, b in configs:
                per_seed = [
                    run_instance(args, m, graph, b, lam, s)
                    for s in range(args.trials)
                ]
                new = summarise(per_seed, m, graph, b, lam)
                rows += new
                pairs += paired_rows(per_seed, m, graph, b, lam)
                for r in new:
                    print(
                        f"lam={lam} {graph:<4} b={b} m={m:<4} "
                        f"{r['ordering']:<11} base={r['base_acc'] * 100:6.2f}% "
                        f"acc={r['acc'] * 100:6.2f}% "
                        f"delta={r['delta'] * 100:+6.2f}% "
                        f"fixed={r['fixed'] * 100:5.2f}% "
                        f"broke={r['broke'] * 100:5.2f}%",
                        flush=True,
                    )

    csv_path = args.prefix.with_suffix(".csv")
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote CSV -> {csv_path}")
    write_markdown(rows, pairs, args.prefix.with_suffix(".md"), args)


if __name__ == "__main__":
    main()
