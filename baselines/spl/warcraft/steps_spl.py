from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "tasks" / "warcraft") not in sys.path:
    sys.path.append(str(ROOT / "tasks" / "warcraft"))

from steps_walk import loop_erase, sequence_to_edges  # noqa: E402

from steps_sdd import StepsMoves, canonical_edge_ids  # noqa: E402

LOG_HALF = float(np.log(0.5))


class StepsSplAdapter:
    """Translate between SPL's per-edge head and the steps SDD's move variables.

    The network scores the 2g(g-1) edges in the repo's label order; every move
    bit crossing edge e takes e's literal weights, and the absorbing t -> t
    bits take a neutral log(1/2) on both literals. A walk therefore scores the
    product of p_e over the edges it crosses and (1 - p_e) over the move bits
    it leaves off, which on a simple path is SPL's per-edge score up to a
    per-board constant.
    """

    def __init__(self, sm: StepsMoves, device="cuda"):
        self.sm = sm
        self.g = sm.g
        self.device = device
        self.num_edges = 2 * sm.g * (sm.g - 1)
        order = np.argsort(sm.var)
        assert np.array_equal(sm.var[order], np.arange(1, sm.num_vars + 1))
        # Move row of SDD variable j + 1.
        self.row_of_var = order
        edge = sm.moves[order, 3]
        self.absorbing = torch.as_tensor(edge < 0, device=device)
        self.edge_of_var = torch.as_tensor(np.where(edge < 0, 0, edge), device=device)
        self.var_of_move = {
            (int(i), int(u), int(v)): int(sm.var[k]) for k, (i, u, v, _) in enumerate(sm.moves)
        }
        self.edge_ends = {e: uv for uv, e in canonical_edge_ids(sm.g).items()}

    def litweights(self, logprobs):
        """Per-variable [negative, positive] log weights from (B, E, K) edge log-probabilities."""
        neg, pos = self.lit_tensors(logprobs)
        return [[n, p] for n, p in zip(neg.unbind(1), pos.unbind(1))]

    def lit_tensors(self, logprobs):
        """(B, V, K) negative and positive literal log weights, by variable id."""
        pos = logprobs[:, self.edge_of_var, :]
        x = -pos
        neg = torch.where(x < 0.6931471805599453, torch.log(-torch.expm1(-x)),
                          torch.log1p(-torch.exp(-x))).clamp(max=-1e-5)
        half = torch.full_like(pos, LOG_HALF)
        mask = self.absorbing.view(1, -1, 1)
        pos = torch.where(mask, half, pos)
        neg = torch.where(mask, half, neg)
        return neg, pos

    def path_cells(self, edge_bits) -> list[int]:
        """The s-t cell sequence of a simple path given as an edge vector."""
        adj: dict[int, list[int]] = {}
        for e in np.nonzero(np.asarray(edge_bits) > 0)[0]:
            a, b = self.edge_ends[int(e)]
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        t = self.g * self.g - 1
        cells, prev = [0], None
        while cells[-1] != t:
            nxt = [c for c in adj[cells[-1]] if c != prev]
            prev = cells[-1]
            cells.append(nxt[0])
        return cells

    def edge_labels_to_bits(self, labels) -> torch.Tensor:
        """(B, E) edge labels -> (B, num_vars) move bits of the t-padded walk, by variable id."""
        labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
        L, t = self.sm.horizon, self.g * self.g - 1
        out = np.zeros((len(labels), self.sm.num_vars), dtype=np.float32)
        for b, lab in enumerate(labels):
            cells = self.path_cells(lab)
            cells = cells + [t] * (L + 1 - len(cells))
            for i in range(L):
                out[b, self.var_of_move[(i, cells[i], cells[i + 1])] - 1] = 1.0
        return torch.as_tensor(out, device=self.device)

    def bits_to_walks(self, bits) -> list[list[int]]:
        """(B, num_vars) move bits by variable id -> the walk's cell sequences."""
        bits = bits.detach().cpu().numpy() if torch.is_tensor(bits) else np.asarray(bits)
        L = self.sm.horizon
        walks = []
        for row in bits:
            on = self.row_of_var[np.nonzero(row > 0)[0]]
            moves = self.sm.moves[on]
            moves = moves[np.argsort(moves[:, 0])]
            assert len(moves) == L and np.array_equal(moves[:, 0], np.arange(L))
            assert np.array_equal(moves[1:, 1], moves[:-1, 2])
            walks.append([int(moves[0, 1])] + [int(v) for v in moves[:, 2]])
        return walks

    def bits_to_edges(self, bits):
        """Move bits -> (B, E) edge predictions after loop erasure, plus cells snipped per board."""
        t = self.g * self.g - 1
        edges, snipped = [], []
        for walk in self.bits_to_walks(bits):
            arrival = walk.index(t)
            erased, n = loop_erase(walk[: arrival + 1])
            edges.append(sequence_to_edges(self.g, erased))
            snipped.append(n)
        return torch.as_tensor(np.stack(edges), device=self.device), snipped
