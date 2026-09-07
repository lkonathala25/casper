from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for sub in ("casper", "tasks/warcraft"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from minedit import MinEditL2  # noqa: E402

from steps_encoding import build_steps_encoding, successors  # noqa: E402


def edge_index_maps(N: int):
    """Canonical edge indexing: horizontal edges row-major, then vertical.

    Matches `edges.npy` and the right_idx/down_idx helpers used by the
    edge-based pipeline, so decoded edge vectors are directly comparable.
    """
    n_w = N - 1
    n_right = N * n_w

    def right_idx(i, j):
        return i * n_w + j

    def down_idx(i, j):
        return n_right + i * N + j

    edges = [None] * (2 * N * (N - 1))
    for i in range(N):
        for j in range(N - 1):
            edges[right_idx(i, j)] = (i, j, i, j + 1)
    for i in range(N - 1):
        for j in range(N):
            edges[down_idx(i, j)] = (i, j, i + 1, j)
    return edges, right_idx, down_idx


def edge_endpoints_array(N: int) -> np.ndarray:
    """(n_edges, 2) flat vertex ids, built exactly as compute_cost_ratio does."""
    edges, _, _ = edge_index_maps(N)
    arr = np.asarray(edges, dtype=np.int64)
    return np.stack([arr[:, 0] * N + arr[:, 1], arr[:, 2] * N + arr[:, 3]], axis=1)


def loop_erase(seq):
    """Erase loops from a cell sequence, snipping each detour on revisit.

    Returns (erased, n_snipped) where n_snipped is how many cells the erasure
    removed from the de-padded walk.
    """
    out: list[int] = []
    pos: dict[int, int] = {}
    for v in seq:
        if v in pos:
            k = pos[v]
            for dead in out[k + 1:]:
                del pos[dead]
            del out[k + 1:]
        else:
            pos[v] = len(out)
            out.append(v)
    return out, len(seq) - len(out)


def sequence_to_edges(N: int, cells) -> np.ndarray:
    """Binary edge vector for an ordered cell sequence under the canonical indexing."""
    _, right_idx, down_idx = edge_index_maps(N)
    vec = np.zeros(2 * N * (N - 1), dtype=np.uint8)
    for a, b in zip(cells, cells[1:]):
        r1, c1 = divmod(a, N)
        r2, c2 = divmod(b, N)
        if r1 == r2:
            vec[right_idx(r1, min(c1, c2))] = 1
        else:
            vec[down_idx(min(r1, r2), c1)] = 1
    return vec


@dataclass
class WalkStats:
    """Per-board bookkeeping for one guided walk."""

    arrival: int
    dead_ends: int = 0
    forced_steps: int = 0
    decisions: int = 0
    ties_at_top: int = 0
    strict_reversals: int = 0
    divergences: list = field(default_factory=list)


class StepsWalkDecoder:
    """Sequential projection over X_0..X_L on the trimmed steps encoding.

    The feasible set at step i is succ(x_{i-1}) cap D_i — at most four cells,
    or {t} alone once the target is reached. Because the trimmed compile
    derives zero resolvents, that set is exactly what the compiled artifact
    licenses; `feasible_matches_artifact` checks this rather than assuming it.
    """

    def __init__(self, g: int, eps: float = 1e-3, dtype=torch.float32):
        self.g = g
        self.eps = eps
        self.dtype = dtype
        self.enc = build_steps_encoding(g, trimmed=True)
        self.domain_sets = [set(d) for d in self.enc.domains]
        self.local = [self.enc.local_index(i) for i in range(self.enc.horizon + 1)]
        self._minedit = {}

    def feasible(self, i: int, prev: int) -> tuple[int, ...]:
        """succ(prev) cap D_i, the candidates for X_i given X_{i-1} = prev."""
        d = self.domain_sets[i]
        return tuple(v for v in successors(self.g, prev) if v in d)

    def _minedit_for(self, n_categories: int) -> MinEditL2:
        if n_categories not in self._minedit:
            self._minedit[n_categories] = MinEditL2(n_categories, eps=self.eps)
        return self._minedit[n_categories]

    def minedit_distances(self, i: int, probs: torch.Tensor, cands) -> list[float]:
        """||q - p||_2 for forcing each candidate to be the argmax at step i.

        This is the full per-candidate computation Improvement 1 replaces: the
        same MinEditL2 the edge-based pipeline uses, at the same eps.
        """
        layer = self._minedit_for(len(self.enc.domains[i]))
        p = probs.unsqueeze(0)
        return [
            float(layer(p, self.local[i][c], True)[1][0]) for c in cands
        ]

    def _break_tie(self, cands, values, best, tie_break, reversal):
        """Pick among the candidates attaining `best`, resolving an exact tie.

        Ties are broken against immediate reversal because a reversal is a
        provably-erased move — loop erasure deletes it downstream by
        construction — so among equally-scored candidates we prefer one that
        survives to the output. Non-tie selections are untouched; the
        Statement's argmax result governs those and is unaffected.
        """
        tied = [c for c, v in zip(cands, values) if v == best]
        if tie_break == "no-backtrack" and len(tied) > 1 and reversal is not None:
            others = [c for c in tied if c != reversal]
            if others:
                tied = others
        return tied[0]

    def walk(self, scores, selection: str = "argmax", early_cut: bool = True,
             tie_break: str = "no-backtrack", forbid_reversal: bool = False):
        """Run the guided walk and return (sequence, WalkStats).

        `scores` is (L+1, n): the tick-score of each cell at each time. Only the
        entries on D_i are read. `selection` is "argmax" (Improvement 1) or
        "minedit" (the full per-candidate distance computation it replaces).
        `early_cut` stops at absorption and fills the rest with t (Improvement 2).
        `tie_break` is "no-backtrack" (the adopted rule, see `_break_tie`) or
        "index", which takes the lowest cell index among tied candidates.

        The tie-break acts only on exact ties at the top score; any strict
        winner is taken regardless of direction, so this is not a ban on
        backward moves. `forbid_reversal` is the stronger, separate rule that
        *is* such a ban: it drops the previously occupied cell from the
        candidate set whenever anything else is available, so a strictly
        higher-scoring reversal is refused too. It is not adopted; it exists so
        the Phase 5 screen can separate decoder-side forward pressure from
        bridge-side forward pressure.
        """
        L, t = self.enc.horizon, self.enc.target
        seq = [self.enc.source]
        stats = WalkStats(arrival=L)
        arrived = False

        for i in range(1, L + 1):
            if arrived and early_cut:
                seq.append(t)
                continue
            cands = self.feasible(i, seq[-1])
            if not cands:
                stats.dead_ends += 1
                seq.append(seq[-1])
                continue
            reversal = seq[-2] if len(seq) > 1 else None
            if forbid_reversal and len(cands) > 1 and reversal is not None:
                trimmed = tuple(c for c in cands if c != reversal)
                if trimmed:
                    cands = trimmed
            if len(cands) == 1:
                stats.forced_steps += 1
                pick = cands[0]
            else:
                stats.decisions += 1
                dom = self.enc.domains[i]
                row = torch.as_tensor(
                    [scores[i][v] for v in dom], dtype=self.dtype
                )
                probs = torch.softmax(row, dim=0)
                cand_probs = [float(probs[self.local[i][c]]) for c in cands]
                top = max(cand_probs)
                n_top = sum(1 for v in cand_probs if v == top)
                if n_top > 1:
                    stats.ties_at_top += 1
                elif reversal is not None and cands[cand_probs.index(top)] == reversal:
                    stats.strict_reversals += 1
                by_argmax = self._break_tie(
                    cands, cand_probs, top, tie_break, reversal
                )
                if selection == "argmax":
                    pick = by_argmax
                elif selection == "minedit":
                    dists = self.minedit_distances(i, probs, cands)
                    pick = self._break_tie(
                        cands, dists, min(dists), tie_break, reversal
                    )
                    if pick != by_argmax:
                        stats.divergences.append(
                            {
                                "step": i,
                                "candidates": list(cands),
                                "argmax": by_argmax,
                                "minedit": pick,
                                "distances": dists,
                                "probs": cand_probs,
                                "reversal": reversal,
                            }
                        )
                else:
                    raise ValueError(f"unknown selection {selection!r}")
            seq.append(pick)
            if pick == t and not arrived:
                arrived = True
                stats.arrival = i

        return seq, stats

    def decode(self, seq):
        """Loop-erase the walk, trim the absorbing padding, and emit an edge vector."""
        t = self.enc.target
        core = list(seq)
        while len(core) > 1 and core[-1] == t and core[-2] == t:
            core.pop()
        erased, snipped = loop_erase(core)
        return erased, snipped, sequence_to_edges(self.g, erased)

    def feasible_matches_artifact(self, compiled) -> bool:
        """Check succ(u) cap D_i against the compiled artifact's own clauses.

        With zero resolvents the compiled per-variable clauses are the input
        clauses, so the walk's feasible set must be recoverable from them.
        """
        by_step = {}
        for pi in compiled.per_var_clauses:
            for clause in pi:
                vs = sorted(lit.var for lit in clause.literals)
                if len(vs) == 1:
                    lo = hi = vs[0]
                else:
                    lo, hi = vs
                by_step.setdefault(lo, []).append((clause, lo, hi))

        for i in range(self.enc.horizon):
            for u in self.enc.domains[i]:
                want = set(self.feasible(i + 1, u))
                got = set(self.enc.domains[i + 1])
                for clause, lo, hi in by_step.get(i, []):
                    if hi != i + 1:
                        continue
                    guard = clause.get(lo)
                    if lo == hi:
                        allowed = {self.enc.domains[hi][k] for k in guard}
                        got &= allowed
                        continue
                    if self.local[i][u] in guard:
                        continue
                    nxt = clause.get(hi) or frozenset()
                    got &= {self.enc.domains[hi][k] for k in nxt}
                if got != want:
                    return False
        return True
