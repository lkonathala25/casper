from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tasks/warcraft"):
    if str(ROOT / sub) not in sys.path:
        sys.path.append(str(ROOT / sub))

from node_categories import build_node_categories  # noqa: E402


class PlainBroadcastBridge:
    """Per-node categorical outputs -> per-tick scores for the steps walk.

    The tick-score of cell v at time i is the network's on-path score for cell
    v, restricted to D_i. No time weighting, no normalization, no lambda
    machinery: the same scalar is broadcast to every tick, which is our reading
    of the reference table's lambda=0 row.

    The head is categorical, not a scalar per cell, so the scalar has to be
    defined. We take P(on-path) = 1 - P(skip). Interior nodes carry a 'skip'
    category (all incident edges off) plus one category per unordered pair of
    incident edges — 7 under the full alphabet at a degree-4 node — so the skip
    probability is exactly the network's belief that the path misses the cell.
    The source and sink have no skip category (every category lights exactly
    one incident edge), so their on-path score is identically 1.0, which is
    also what the encoding asserts by pinning D_0 and D_L.
    """

    def __init__(self, grid_size: int = 12, alphabet: str = "full"):
        self.grid_size = grid_size
        self.alphabet = alphabet
        self.node_info, self.var_ranges, self.node_order, self.total_dim = (
            build_node_categories(grid_size, alphabet=alphabet)
        )
        offsets, off = [], 0
        skip_slots = []
        for node in self.node_order:
            offsets.append(off)
            cats = self.node_info[node]["categories"]
            skip_slots.append(off + cats.index("skip") if "skip" in cats else -1)
            off += len(cats)
        self.offsets = offsets
        self.skip_slots = np.asarray(skip_slots, dtype=np.int64)

    @property
    def n_cells(self) -> int:
        return self.grid_size * self.grid_size

    def cell_scores(self, probs) -> np.ndarray:
        """(B, total_dim) per-node categorical probabilities -> (B, n) on-path scores.

        Cells whose node has no skip category (source, sink) score 1.0.
        """
        p = probs.detach().cpu().numpy() if torch.is_tensor(probs) else np.asarray(probs)
        if p.ndim == 1:
            p = p[None, :]
        out = np.ones((p.shape[0], self.n_cells), dtype=np.float64)
        has_skip = self.skip_slots >= 0
        out[:, has_skip] = 1.0 - p[:, self.skip_slots[has_skip]]
        return out

    def tick_scores(self, cell_score, horizon: int) -> np.ndarray:
        """Broadcast one board's (n,) cell scores across ticks 0..horizon."""
        cell_score = np.asarray(cell_score, dtype=np.float64)
        return np.broadcast_to(cell_score[None, :], (horizon + 1, cell_score.size))

    def gt_cell_scores(self, edge_labels, edges_lookup) -> np.ndarray:
        """(B, n_edges) ground-truth edge vectors -> (B, n) scores, 1.0 on-path.

        Used by the pass-through gate: perfect information in, so the pipeline
        must return the ground-truth path itself.
        """
        e = np.asarray(edge_labels).astype(bool)
        lut = np.asarray(edges_lookup, dtype=np.int64)
        a = lut[:, 0] * self.grid_size + lut[:, 1]
        b = lut[:, 2] * self.grid_size + lut[:, 3]
        out = np.zeros((e.shape[0], self.n_cells), dtype=np.float64)
        for i in range(e.shape[0]):
            on = e[i]
            out[i, np.unique(np.concatenate([a[on], b[on]]))] = 1.0
        return out


def softmax_per_node(logits: torch.Tensor, var_ranges) -> torch.Tensor:
    """Softmax within each node's own category block."""
    out = torch.zeros_like(logits)
    off = 0
    for vr in var_ranges:
        out[:, off:off + vr] = torch.softmax(logits[:, off:off + vr], dim=1)
        off += vr
    return out
