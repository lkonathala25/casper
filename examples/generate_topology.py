from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signed_clauses import ConstraintSet, SignedClause, SignedLiteral  # noqa: E402


def class_means(n_classes: int, radius: float = 10.0) -> np.ndarray:
    """K class means placed evenly on a circle of the given radius."""
    angles = 2.0 * np.pi * np.arange(n_classes) / n_classes
    return radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def generate_data(
    m: int,
    n_classes: int,
    theta: float = 1.0,
    radius: float = 10.0,
    lam: float = 0.5,
    eps: float = 1.0,
    noise_frac: float = 0.5,
    noise_dist: str = "gaussian",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw chi = <X_1..X_m> from K isotropic Gaussians on a circle.

    Labels are uniform over the K classes; X_i ~ N(mu_{c_i}, theta^2 I). A
    `noise_frac` fraction of the points is then corrupted by convex mixing,
    X_i <- (1-lam) X_i + lam * e. `noise_dist` selects the corruption law:
    "gaussian" draws e ~ N(0, eps^2 I), "uniform" draws e ~ eps * U[0,1]^2.
    The class-conditional scatter stays Gaussian either way. Returns (X, labels)
    with X of shape (m, 2).
    """
    rng = np.random.default_rng() if rng is None else rng
    mus = class_means(n_classes, radius)
    labels = rng.integers(0, n_classes, size=m)
    X = mus[labels] + theta * rng.standard_normal((m, 2))

    n_noisy = int(round(noise_frac * m))
    if n_noisy > 0 and lam > 0.0:
        idx = rng.choice(m, size=n_noisy, replace=False)
        if noise_dist == "gaussian":
            noise = eps * rng.standard_normal((n_noisy, 2))
        elif noise_dist == "uniform":
            noise = eps * rng.random((n_noisy, 2))
        else:
            raise ValueError(f"unknown noise_dist {noise_dist!r}")
        X[idx] = (1.0 - lam) * X[idx] + lam * noise
    return X, labels


def banded_edges(m: int, b: int) -> list[tuple[int, int]]:
    """Edge set of the `exp` topology: {i,j} in E iff 0 < |i-j| <= b."""
    return [
        (i, j)
        for i in range(m)
        for j in range(i + 1, min(i + b + 1, m))
    ]


def tree_edges(
    m: int, rng: random.Random, shape: str = "random"
) -> list[tuple[int, int]]:
    """Edge set of the `poly` topology: an undirected spanning tree, m-1 edges.

    `shape` is "random" (each node attaches to a uniform earlier node), "star"
    (every node attaches to node 0) or "path" (node i attaches to i-1).
    """
    if shape == "star":
        return [(0, i) for i in range(1, m)]
    if shape == "path":
        return [(i - 1, i) for i in range(1, m)]
    if shape != "random":
        raise ValueError(f"unknown tree shape {shape!r}")
    return [(rng.randrange(i), i) for i in range(1, m)]


def build_edges(
    m: int,
    graph: str,
    b: int = 1,
    rng: random.Random | None = None,
    tree_shape: str = "random",
) -> list[tuple[int, int]]:
    """Dispatch to the `exp` (banded) or `poly` (spanning tree) edge scheme."""
    if graph == "exp":
        return banded_edges(m, b)
    if graph == "poly":
        return tree_edges(m, rng or random.Random(0), tree_shape)
    raise ValueError(f"unknown graph type {graph!r}")


def _sample_values(
    n_classes: int, size: int, rng: random.Random, forced: int | None = None
) -> frozenset[int]:
    """Sample a value-set S of `size` classes, optionally containing `forced`."""
    pool = [v for v in range(n_classes) if v != forced]
    take = size if forced is None else size - 1
    chosen = rng.sample(pool, take) if take > 0 else []
    if forced is not None:
        chosen.append(forced)
    return frozenset(chosen)


def build_constraints(
    m: int,
    n_classes: int,
    edges: list[tuple[int, int]],
    labels: np.ndarray | None = None,
    tau: float = 0.5,
    clauses_per_edge: int = 1,
    rng: random.Random | None = None,
) -> ConstraintSet:
    """Build one ConstraintSet of binary signed clauses over the edge set.

    Each edge {X_i, X_j} contributes `clauses_per_edge` clauses of the form
    V_t(S_i : Y_i) or V_t(S_j : Y_j) with |S| = round(tau * K), clipped to
    [1, K-1] so no literal is tautological. When `labels` is given, one endpoint
    of every clause is forced to contain its ground-truth class, which keeps the
    instance satisfiable (the compile then measures real elimination work rather
    than an early UNSAT).
    """
    rng = random.Random(0) if rng is None else rng
    size = min(max(int(round(tau * n_classes)), 1), n_classes - 1)

    clauses: list[SignedClause] = []
    for i, j in edges:
        for _ in range(clauses_per_edge):
            if labels is None:
                s_i = _sample_values(n_classes, size, rng)
                s_j = _sample_values(n_classes, size, rng)
            else:
                pin = i if rng.random() < 0.5 else j
                s_i = _sample_values(
                    n_classes, size, rng,
                    forced=int(labels[i]) if pin == i else None,
                )
                s_j = _sample_values(
                    n_classes, size, rng,
                    forced=int(labels[j]) if pin == j else None,
                )
            clauses.append(SignedClause.from_literals([
                SignedLiteral(i, s_i), SignedLiteral(j, s_j),
            ]))

    return ConstraintSet(
        var_names=[f"y_{i}" for i in range(m)],
        var_domains=[n_classes] * m,
        clauses=clauses,
        ordering=list(range(m)),
    )


def constraint_set_to_dsl(cs: ConstraintSet) -> str:
    """Render a ConstraintSet back to the parseable signed-clause DSL."""
    lines = [f"var {n} {h}" for n, h in zip(cs.var_names, cs.var_domains)]
    lines.append("")
    lines.append("ordering " + " ".join(cs.var_names[i] for i in cs.ordering))
    lines.append("")
    for c in cs.clauses:
        lines.append(" or ".join(
            "[" + ",".join(str(v + cs.value_base) for v in sorted(lit.values))
            + f"]:{cs.var_names[lit.var]}"
            for lit in c.literals
        ))
    return "\n".join(lines) + "\n"


def generate_instance(
    m: int,
    graph: str,
    n_classes: int = 4,
    b: int = 1,
    tau: float = 0.5,
    clauses_per_edge: int = 1,
    theta: float = 1.0,
    radius: float = 10.0,
    lam: float = 0.5,
    eps: float = 1.0,
    noise_frac: float = 0.5,
    noise_dist: str = "gaussian",
    tree_shape: str = "random",
    seed: int = 0,
    satisfiable: bool = True,
) -> tuple[ConstraintSet, np.ndarray, np.ndarray]:
    """Generate one (constraints, X, labels) instance for a topology config."""
    np_rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    X, labels = generate_data(
        m, n_classes, theta=theta, radius=radius, lam=lam, eps=eps,
        noise_frac=noise_frac, noise_dist=noise_dist, rng=np_rng,
    )
    edges = build_edges(m, graph, b=b, rng=py_rng, tree_shape=tree_shape)
    cs = build_constraints(
        m, n_classes, edges, labels=labels if satisfiable else None,
        tau=tau, clauses_per_edge=clauses_per_edge, rng=py_rng,
    )
    return cs, X, labels


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate a synthetic topology instance (data + DSL)."
    )
    p.add_argument("m", type=int, help="Number of variables/points.")
    p.add_argument("graph", choices=("exp", "poly"))
    p.add_argument("out_path", type=Path, help="Where to write the DSL.")
    p.add_argument("--classes", type=int, default=4, help="K.")
    p.add_argument("--band", type=int, default=1, help="Bandwidth b for `exp`.")
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--clauses-per-edge", type=int, default=1)
    p.add_argument("--theta", type=float, default=1.0)
    p.add_argument("--radius", type=float, default=10.0)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--noise-frac", type=float, default=0.5)
    p.add_argument("--tree-shape", choices=("random", "star", "path"),
                   default="random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-data", type=Path, default=None,
                   help="Optional .npz for the generated points and labels.")
    args = p.parse_args()

    cs, X, labels = generate_instance(
        args.m, args.graph, n_classes=args.classes, b=args.band, tau=args.tau,
        clauses_per_edge=args.clauses_per_edge, theta=args.theta,
        radius=args.radius, lam=args.lam, eps=args.eps,
        noise_frac=args.noise_frac, tree_shape=args.tree_shape, seed=args.seed,
    )
    args.out_path.write_text(constraint_set_to_dsl(cs))
    if args.save_data is not None:
        np.savez(args.save_data, X=X, labels=labels)
    print(
        f"m={args.m} graph={args.graph} K={args.classes} "
        f"clauses={len(cs.clauses)} -> {args.out_path}"
    )


if __name__ == "__main__":
    main()
