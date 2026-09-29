from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "tasks" / "warcraft") not in sys.path:
    sys.path.append(str(ROOT / "tasks" / "warcraft"))

from steps_encoding import successors, trimmed_domain  # noqa: E402

DEFAULT_OUT = ROOT / "constraints" / "warcraft" / "spl_steps"


def canonical_edge_ids(g: int) -> dict[tuple[int, int], int]:
    """(min cell, max cell) -> edge index in the repo's label order.

    Horizontal edges row-major, then vertical, as in steps_walk.edge_index_maps
    and the *_edge_labels.npy files.
    """
    ids = {}
    for r in range(g):
        for c in range(g - 1):
            ids[(r * g + c, r * g + c + 1)] = r * (g - 1) + c
    for r in range(g - 1):
        for c in range(g):
            ids[(r * g + c, (r + 1) * g + c)] = g * (g - 1) + r * g + c
    return ids


@dataclass
class StepsMoves:
    """The legal moves of the trimmed steps encoding, one Boolean SDD variable each.

    Row k of `moves` is (tick, from cell, to cell, edge), edge = -1 for the
    absorbing t -> t move. `var` is the SDD variable id of each row. Parity
    fixes the direction an edge can be crossed at a given tick, so (tick, edge)
    already identifies a non-absorbing move.
    """

    g: int
    moves: np.ndarray
    var: np.ndarray

    @property
    def horizon(self) -> int:
        return self.g * self.g - 1

    @property
    def num_vars(self) -> int:
        return len(self.moves)

    def ticks(self) -> list[np.ndarray]:
        """Row indices of the moves at each tick."""
        return [np.nonzero(self.moves[:, 0] == i)[0] for i in range(self.horizon)]


def enumerate_moves(g: int) -> StepsMoves:
    """Legal moves per tick, with variable 1 assigned to the last absorbing move.

    pypsdd's ff_ll reads the positive weight of literal +1 from the negative
    slot (`literal > 1`). Every model forces the final t -> t move and it
    carries a neutral weight, so putting it on variable 1 makes that harmless.
    """
    n = g * g
    L, t = n - 1, n - 1
    edge_id = canonical_edge_ids(g)
    domains = [trimmed_domain(g, i) for i in range(L + 1)]
    rows = []
    for i in range(L):
        nxt = set(domains[i + 1])
        for u in domains[i]:
            for v in successors(g, u):
                if v in nxt:
                    e = -1 if u == v == t else edge_id[(min(u, v), max(u, v))]
                    rows.append((i, u, v, e))
    moves = np.asarray(rows, dtype=np.int64)
    last = len(moves) - 1
    assert tuple(moves[last]) == (L - 1, t, t, -1)
    var = np.empty(len(moves), dtype=np.int64)
    var[last] = 1
    var[:last] = np.arange(2, len(moves) + 1)
    return StepsMoves(g=g, moves=moves, var=var)


def _balanced(leaves: list[int], next_id: list[int], lines: list[str]) -> int:
    """Append a balanced vtree over `leaves` (variable ids) and return its root id."""
    if len(leaves) == 1:
        nid = next_id[0]
        next_id[0] += 1
        lines.append(f"L {nid} {leaves[0]}")
        return nid
    mid = len(leaves) // 2
    left = _balanced(leaves[:mid], next_id, lines)
    right = _balanced(leaves[mid:], next_id, lines)
    nid = next_id[0]
    next_id[0] += 1
    lines.append(f"I {nid} {left} {right}")
    return nid


def write_vtree(sm: StepsMoves, path: Path) -> None:
    """Vtree: a balanced block per tick, the blocks chained right-linearly in time."""
    lines: list[str] = []
    next_id = [0]
    blocks = [_balanced([int(sm.var[k]) for k in rows], next_id, lines)
              for rows in sm.ticks()]
    right = blocks[-1]
    for block in reversed(blocks[:-1]):
        nid = next_id[0]
        next_id[0] += 1
        lines.append(f"I {nid} {block} {right}")
        right = nid
    path.write_text(f"vtree {next_id[0]}\n" + "\n".join(lines) + "\n")


def compile_steps_sdd(sm: StepsMoves, vtree_path: Path):
    """SDD of 'the move bits trace an absorbing s-t walk', built backwards over time.

    G_i(v) is 'from cell v at tick i the remaining moves form a legal walk':
    the disjunction over v's moves m = (v -> w) of 'exactly m at tick i' and
    G_{i+1}(w). Returns (manager, root).
    """
    from pysdd.sdd import SddManager, Vtree

    mgr = SddManager.from_vtree(Vtree.from_file(str(vtree_path).encode()))
    mgr.auto_gc_and_minimize_off()
    L, t = sm.horizon, sm.g * sm.g - 1
    reach = {t: mgr.true()}
    for i in reversed(range(L)):
        rows = sm.ticks()[i]
        none = mgr.true()
        for k in rows:
            none = none & mgr.literal(-int(sm.var[k]))
        by_from: dict[int, object] = {}
        for k in rows:
            _, u, v, _ = sm.moves[k]
            if int(v) not in reach:
                continue
            x = int(sm.var[k])
            exactly = none.condition(-x) & mgr.literal(x)
            term = exactly & reach[int(v)]
            by_from[int(u)] = by_from[int(u)] | term if int(u) in by_from else term
        reach = by_from
    return mgr, reach[0]


def build(g: int, out_dir: Path = DEFAULT_OUT) -> dict:
    """Compile and save steps_{g}.sdd/.vtree plus the variable -> move table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sm = enumerate_moves(g)
    vtree_path = out_dir / f"steps_{g}.vtree"
    sdd_path = out_dir / f"steps_{g}.sdd"
    write_vtree(sm, vtree_path)
    t0 = time.time()
    mgr, root = compile_steps_sdd(sm, vtree_path)
    elapsed = time.time() - t0
    root.save(str(sdd_path).encode())
    np.savez(out_dir / f"steps_{g}_moves.npz", moves=sm.moves, var=sm.var, g=g)
    summary = {
        "g": g,
        "vars": sm.num_vars,
        "sdd_size": int(root.size()),
        "sdd_count": int(root.count()),
        "compile_s": round(elapsed, 2),
        "file_mb": round(sdd_path.stat().st_size / 1e6, 2),
    }
    if g <= 4:
        summary["model_count"] = int(root.global_model_count())
    (out_dir / f"steps_{g}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_moves(g: int, out_dir: Path = DEFAULT_OUT) -> StepsMoves:
    """The variable -> move table saved beside a compiled steps SDD."""
    z = np.load(out_dir / f"steps_{g}_moves.npz")
    return StepsMoves(g=int(z["g"]), moves=z["moves"], var=z["var"])


def main():
    """Compile the steps SDD for the requested grid sizes."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, nargs="+", default=[12])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    for g in args.size:
        print(json.dumps(build(g, args.out)))


if __name__ == "__main__":
    main()
