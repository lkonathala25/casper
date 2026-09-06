from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


def cell_rc(g: int, v: int) -> tuple[int, int]:
    """Row-major (row, col) of cell index `v` on a g x g board."""
    return divmod(v, g)


def manhattan(g: int, a: int, b: int) -> int:
    """Graph distance d(a, b) on the four-neighbour g x g grid."""
    ra, ca = cell_rc(g, a)
    rb, cb = cell_rc(g, b)
    return abs(ra - rb) + abs(ca - cb)


def neighbours(g: int, v: int) -> tuple[int, ...]:
    """N(v): the up/down/left/right cells of `v`, ascending."""
    r, c = cell_rc(g, v)
    out = []
    for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < g and 0 <= cc < g:
            out.append(rr * g + cc)
    return tuple(sorted(out))


def successors(g: int, v: int, *, waiting: bool = False) -> tuple[int, ...]:
    """succ(v): N(v) for v != t, and {t} for v = t (absorption).

    With `waiting`, every cell may also repeat itself; this is the untrimmed
    control variant of the tex table, where the walk needs an explicit waiting
    move to cover arrival times shorter than the horizon.
    """
    t = g * g - 1
    if v == t:
        return (t,)
    nb = neighbours(g, v)
    return tuple(sorted(set(nb) | {v})) if waiting else nb


def trimmed_domain(g: int, i: int) -> tuple[int, ...]:
    """D_i under the forward, parity and backward tests of the tex."""
    n = g * g
    L = n - 1
    s, t = 0, n - 1
    out = [
        v
        for v in range(n)
        if v != t
        and manhattan(g, s, v) <= i
        and i % 2 == manhattan(g, s, v) % 2
        and manhattan(g, v, t) <= L - i
    ]
    if i >= manhattan(g, s, t):
        out.append(t)
    return tuple(sorted(out))


@dataclass(frozen=True)
class StepsClause:
    """One unrolled conjunct [guard]:X_step or [guard]:X_step or [succ]:X_next.

    Value sets are global cell indices, exactly as the tex prints them; the
    per-variable local re-indexing happens only when the DSL text is emitted.
    An empty `guard` means D_step was a singleton and the guard was dropped,
    leaving a unary clause on X_{step+1}.
    """

    step: int
    guard: tuple[int, ...]
    succ: tuple[int, ...]

    def __repr__(self) -> str:
        rhs = "[" + ",".join(str(v) for v in self.succ) + f"]:X_{self.step + 1}"
        if not self.guard:
            return rhs
        lhs = "[" + ",".join(str(v) for v in self.guard) + f"]:X_{self.step}"
        return f"{lhs} or {rhs}"


@dataclass(frozen=True)
class StepsEncoding:
    """The time-unrolled walk formula over X_0..X_L for one g x g board."""

    g: int
    trimmed: bool
    waiting: bool
    domains: tuple[tuple[int, ...], ...]
    clauses: tuple[StepsClause, ...]

    @property
    def n(self) -> int:
        return self.g * self.g

    @property
    def horizon(self) -> int:
        return self.n - 1

    @property
    def source(self) -> int:
        return 0

    @property
    def target(self) -> int:
        return self.n - 1

    def local_index(self, step: int) -> dict[int, int]:
        """Cell index -> position within D_step, the DSL value for that step."""
        return {v: k for k, v in enumerate(self.domains[step])}

    def satisfies(self, assignment: Sequence[int]) -> bool:
        """True iff `assignment` lies in D_0 x ... x D_L and satisfies every clause."""
        if len(assignment) != self.horizon + 1:
            return False
        for i, x in enumerate(assignment):
            if x not in self.domains[i]:
                return False
        for c in self.clauses:
            if assignment[c.step] in c.guard:
                continue
            if assignment[c.step + 1] in c.succ:
                continue
            return False
        return True

    def models(self) -> Iterator[tuple[int, ...]]:
        """Enumerate every model by depth-first search over the trimmed domains."""
        L = self.horizon
        by_step: list[list[StepsClause]] = [[] for _ in range(L)]
        for c in self.clauses:
            by_step[c.step].append(c)

        def walk(prefix: list[int]) -> Iterator[tuple[int, ...]]:
            i = len(prefix) - 1
            if i == L:
                yield tuple(prefix)
                return
            for v in self.domains[i + 1]:
                prefix.append(v)
                if all(
                    prefix[c.step] in c.guard or v in c.succ for c in by_step[i]
                ):
                    yield from walk(prefix)
                prefix.pop()

        for v0 in self.domains[0]:
            yield from walk([v0])


def build_steps_encoding(g: int, trimmed: bool = True) -> StepsEncoding:
    """Build the time-unrolled walk encoding for a g x g grid.

    Variables X_0..X_L with L = g^2 - 1, cells numbered row-major with s = 0
    and t = g^2 - 1. `trimmed` declares the distance/parity-trimmed domains D_i
    of the tex; otherwise every X_i gets the full vertex set (endpoints still
    pinned as domain facts) and the waiting move is enabled, which is the
    untrimmed control of the tex table. No AllDifferent clauses are emitted:
    simplicity is recovered downstream by loop erasure.
    """
    if g < 2:
        raise ValueError(f"grid size must be at least 2, got {g}")
    n = g * g
    L = n - 1
    s, t = 0, n - 1
    waiting = not trimmed

    if trimmed:
        domains = tuple(trimmed_domain(g, i) for i in range(L + 1))
    else:
        full = tuple(range(n))
        domains = tuple(
            (s,) if i == 0 else (t,) if i == L else full for i in range(L + 1)
        )

    clauses: list[StepsClause] = []
    for i in range(L):
        d_i, d_next = domains[i], set(domains[i + 1])
        singleton = len(d_i) == 1
        for u in d_i:
            succ = tuple(
                v for v in successors(g, u, waiting=waiting) if v in d_next
            )
            guard = () if singleton else tuple(v for v in d_i if v != u)
            clauses.append(StepsClause(step=i, guard=guard, succ=succ))

    return StepsEncoding(
        g=g,
        trimmed=trimmed,
        waiting=waiting,
        domains=domains,
        clauses=tuple(clauses),
    )


def to_dsl(enc: StepsEncoding) -> str:
    """Render the encoding as a constraint DSL file.

    Each X_i is declared with |D_i| values and its clause value sets are
    re-indexed to 0..|D_i|-1, so the trimmed domains are structural facts of the
    declaration rather than extra unary clauses. The step -> cell map is emitted
    as comments and duplicated in the sidecar JSON of `write_artifacts`.
    """
    L = enc.horizon
    lines = [
        f"# steps encoding: g={enc.g} n={enc.n} L={L} "
        f"s={enc.source} t={enc.target}",
        f"# trimmed={enc.trimmed} waiting={enc.waiting} allow_cycles=True",
        "# X_i values are positions within D_i; see the domain map below.",
        "",
    ]
    for i in range(L + 1):
        lines.append(f"var X_{i} {len(enc.domains[i])}")
    lines.append("")
    lines.append("ordering " + " ".join(f"X_{i}" for i in range(L + 1)))
    lines.append("")
    for i in range(L + 1):
        cells = ",".join(str(v) for v in enc.domains[i])
        lines.append(f"# D_{i} = [{cells}]")
    lines.append("")

    maps = [enc.local_index(i) for i in range(L + 1)]
    for c in enc.clauses:
        rhs = ",".join(str(maps[c.step + 1][v]) for v in c.succ)
        rhs_lit = f"[{rhs}]:X_{c.step + 1}"
        if not c.guard:
            lines.append(rhs_lit)
        else:
            lhs = ",".join(str(maps[c.step][v]) for v in c.guard)
            lines.append(f"[{lhs}]:X_{c.step} or {rhs_lit}")
    lines.append("")
    return "\n".join(lines)


def write_artifacts(enc: StepsEncoding, dsl_path: Path) -> tuple[Path, Path]:
    """Write the DSL file and a sidecar JSON holding the local-to-cell maps."""
    dsl_path = Path(dsl_path)
    dsl_path.parent.mkdir(parents=True, exist_ok=True)
    dsl_path.write_text(to_dsl(enc))
    map_path = dsl_path.with_suffix(".domains.json")
    map_path.write_text(
        json.dumps(
            {
                "g": enc.g,
                "n": enc.n,
                "horizon": enc.horizon,
                "source": enc.source,
                "target": enc.target,
                "trimmed": enc.trimmed,
                "waiting": enc.waiting,
                "n_clauses": len(enc.clauses),
                "domains": [list(d) for d in enc.domains],
            }
        )
    )
    return dsl_path, map_path
