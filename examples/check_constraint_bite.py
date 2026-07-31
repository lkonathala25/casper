from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from constraints import Constraints  # noqa: E402
from resolution import _heuristic_elimination_ordering, compile_constraints  # noqa: E402

from generate_topology import class_means, generate_instance  # noqa: E402


def class_scores(X: np.ndarray, mus: np.ndarray, temp: float) -> np.ndarray:
    """Per-variable class distribution, standing in for a classifier's output.

    Softmax over the negative squared distance to each class centroid. This is
    what the constraint layer receives; the noise-mixing parameters lambda and
    epsilon determine how often its argmax disagrees with the true label.
    """
    d2 = ((X[:, None, :] - mus[None, :, :]) ** 2).sum(axis=2)
    logits = -d2 / temp
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def infer_walk(cons: Constraints, scores: np.ndarray) -> tuple[dict[int, int], int, int]:
    """Assign every variable greedily, highest-scoring feasible value first.

    Returns the assignment, the number of steps where the unconstrained argmax
    was ruled out by the feasible set (the "bite"), and the summed feasible-set
    size across steps.
    """
    assignment: dict[int, int] = {}
    bites = 0
    total_feasible = 0
    for step, var in enumerate(cons.ordering):
        feasible = cons.feasible_set(step, assignment)
        total_feasible += len(feasible)
        raw = int(np.argmax(scores[var]))
        if raw not in feasible:
            bites += 1
        best = max(feasible, key=lambda v: scores[var][v])
        assignment[var] = int(best)
    return assignment, bites, total_feasible


def run_one(args, tau: float, cpe: int, lam: float, seed: int) -> dict:
    """Compile one instance and measure bite plus constrained accuracy."""
    cs, X, labels = generate_instance(
        args.m, args.graph, n_classes=args.classes, b=args.band, tau=tau,
        clauses_per_edge=cpe, theta=args.theta, radius=args.radius, lam=lam,
        eps=args.eps, noise_frac=args.noise_frac, seed=seed,
    )
    mus = class_means(args.classes, args.radius)
    scores = class_scores(X, mus, args.temp)

    cs.ordering = _heuristic_elimination_ordering(cs, "min-fill")
    cons = Constraints.from_compiled(compile_constraints(cs, verbose=False))

    assignment, bites, total_feasible = infer_walk(cons, scores)

    raw_pred = scores.argmax(axis=1)
    n = args.m
    base_acc = float((raw_pred == labels).mean())
    con_pred = np.array([assignment[i] for i in range(n)])
    con_acc = float((con_pred == labels).mean())

    changed = con_pred != raw_pred
    fixed = int(((con_pred == labels) & (raw_pred != labels) & changed).sum())
    broke = int(((con_pred != labels) & (raw_pred == labels) & changed).sum())

    return {
        "bite": bites / n,
        "mean_feasible": total_feasible / n,
        "base_acc": base_acc,
        "con_acc": con_acc,
        "changed": int(changed.sum()) / n,
        "fixed": fixed / n,
        "broke": broke / n,
        "clauses": len(cs.clauses),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Check whether the compiled constraints actually bite: how "
                    "often the feasible set rules out the raw argmax."
    )
    p.add_argument("--m", type=int, default=50)
    p.add_argument("--graph", choices=("exp", "poly"), default="exp")
    p.add_argument("--band", type=int, default=2)
    p.add_argument("--classes", type=int, default=6)
    p.add_argument("--taus", type=float, nargs="+", default=[0.17, 0.33, 0.5])
    p.add_argument("--clauses-per-edge", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--lams", type=float, nargs="+", default=[0.5, 0.9, 1.0])
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--noise-frac", type=float, default=0.5)
    p.add_argument("--theta", type=float, default=1.0)
    p.add_argument("--radius", type=float, default=10.0)
    p.add_argument("--temp", type=float, default=10.0)
    p.add_argument("--trials", type=int, default=5)
    args = p.parse_args()

    print(f"{args.graph} b={args.band} m={args.m} K={args.classes} "
          f"eps={args.eps} noise_frac={args.noise_frac} temp={args.temp}, "
          f"{args.trials} seeds per row")
    print()
    print("lambda   tau  cls/edge  clauses   base acc   bite   mean|S|   "
          "con acc    delta   fixed   broke")
    for lam in args.lams:
        for tau in args.taus:
            for cpe in args.clauses_per_edge:
                res = [run_one(args, tau, cpe, lam, s) for s in range(args.trials)]
                agg = {k: statistics.mean(r[k] for r in res) for k in res[0]}
                delta = agg["con_acc"] - agg["base_acc"]
                print(
                    f"{lam:6.2f} {tau:5.2f} {cpe:9d} {agg['clauses']:8.0f} "
                    f"{agg['base_acc']:10.3f} {agg['bite']:6.3f} "
                    f"{agg['mean_feasible']:8.2f} {agg['con_acc']:9.3f} "
                    f"{delta:+8.3f} {agg['fixed']:7.3f} {agg['broke']:7.3f}"
                )


if __name__ == "__main__":
    main()
