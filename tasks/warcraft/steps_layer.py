from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tasks/warcraft", "tasks/warcraft"):
    if str(ROOT / sub) not in sys.path:
        sys.path.append(str(ROOT / sub))

from edge_bridge import CategoricalEdgeBridge  # noqa: E402
from masked_loss import vectorised_positive_head  # noqa: E402

from steps_bridge import PlainBroadcastBridge, TimeAwareBridge  # noqa: E402
from steps_walk import StepsWalkDecoder, sequence_to_edges  # noqa: E402


class StepsProjectionLayer(nn.Module):
    """Sequential min-edit projection over the steps encoding.

    Distribution in, corrected distribution out, same shape — so the L2
    self-distillation term and `proj_exact` are well defined against it.

    The chain is: per-node categorical probs -> forward bridge (per-cell
    on-path score, then per-tick scores) -> guided walk over the compiled
    trimmed domains, whose feasible set at tick i is succ(x_{i-1}) cap D_i ->
    loop erasure -> edge set -> reverse bridge -> per-node target category ->
    per-node min-edit onto that target.

    **Reverse bridge construction.** The erased walk is a simple s-t path, so
    each node's incident on-edges are determined, and under the full alphabet
    exactly one category matches that pattern (nodes off the path match
    `skip`). That target category is recovered with
    `CategoricalEdgeBridge.edges_to_categories`, the same fail-loud converter
    the edge pipeline uses. The emitted distribution is *not* one-hot: each
    node's original distribution is minimally edited so its argmax becomes the
    target, by the same `vectorised_positive_head` the lazy-exact layer
    applies. Mass is conserved per node block and untouched categories keep
    their probability, so the projection stays a minimal correction rather
    than a hard relabel. There is no "soft where mass is spread" branch — the
    walk commits to one cell per tick, so the target is always unique; the
    softness lives in the min-edit, exactly as it does in the edge pipeline.

    There is no ring-repair loop to disable: the steps encoding admits cyclic
    walks by construction and loop erasure removes them in a single
    deterministic pass. `certify=False` (the training-time default) skips only
    the independent validity check, which is inference-only dressing.
    """

    def __init__(self, grid_size=12, alphabet="full", eps=1e-3,
                 bridge="time-aware", forbid_reversal=True, certify=False,
                 compiled_artifact=None):
        super().__init__()
        self.grid_size = grid_size
        self.eps = eps
        self.certify = certify
        self.forbid_reversal = forbid_reversal
        self.bridge_name = bridge

        self.score_bridge = (
            TimeAwareBridge(grid_size, alphabet=alphabet) if bridge == "time-aware"
            else PlainBroadcastBridge(grid_size, alphabet=alphabet)
        )
        self.edge_bridge = CategoricalEdgeBridge(grid_size, alphabet=alphabet)
        self.decoder = StepsWalkDecoder(grid_size, eps=eps,
                                        compiled_artifact=compiled_artifact)
        self.from_artifact = self.decoder.from_artifact

        self.var_ranges = self.score_bridge.var_ranges
        self.total_dim = self.score_bridge.total_dim
        self.num_edges = 2 * grid_size * (grid_size - 1)

        self._groups = defaultdict(list)
        off = 0
        for n_idx, vr in enumerate(self.var_ranges):
            self._groups[vr].append((n_idx, off))
            off += vr

        self.lifetime_boards = 0
        self.lifetime_dead_ends = 0
        self.lifetime_uncertified = 0

    def push_to_categories(self, probs, target_cats):
        """Min-edit each node's distribution so its argmax becomes its target."""
        adjusted = probs.clone()
        targets = torch.as_tensor(target_cats, dtype=torch.long,
                                  device=probs.device)
        B = probs.shape[0]
        n_edits = 0
        for vr, node_list in self._groups.items():
            slices = [adjusted[:, off:off + vr] for _, off in node_list]
            tgts = [targets[:, n_idx] for n_idx, _ in node_list]
            p_adj, _, n_active = vectorised_positive_head(
                torch.cat(slices, dim=0), torch.cat(tgts, dim=0), eps=self.eps)
            n_edits += int((n_active > 0).sum())
            for k, (_, off) in enumerate(node_list):
                adjusted[:, off:off + vr] = p_adj[k * B:(k + 1) * B]
        return adjusted, n_edits

    def project_edges(self, cell_scores):
        """(B, n_cells) on-path scores -> (B, num_edges) binary, plus per-board logs."""
        L = self.decoder.enc.horizon
        B = cell_scores.shape[0]
        edge_assign = np.zeros((B, self.num_edges), dtype=np.int64)
        logs = []
        for b in range(B):
            ticks = self.score_bridge.tick_scores(cell_scores[b], L)
            seq, stats = self.decoder.walk(
                ticks, selection="argmax", early_cut=True,
                tie_break="no-backtrack", forbid_reversal=self.forbid_reversal)
            erased, snipped, vec = self.decoder.decode(seq)
            edge_assign[b] = vec
            logs.append({"arrival": stats.arrival, "dead_ends": stats.dead_ends,
                         "snipped": snipped, "cells": len(erased)})
        return edge_assign, logs

    def forward(self, probs):
        """Project a batch of per-node categorical distributions."""
        cell_scores = self.score_bridge.cell_scores(probs)
        edge_assign, logs = self.project_edges(cell_scores)

        target_cats = self.edge_bridge.edges_to_categories(edge_assign)
        adjusted, n_edits = self.push_to_categories(probs, target_cats)

        B = probs.shape[0]
        dead = sum(lg["dead_ends"] for lg in logs)
        snipped = [lg["snipped"] for lg in logs]
        self.lifetime_boards += B
        self.lifetime_dead_ends += dead

        info = {
            "edge_assignment": torch.as_tensor(edge_assign),
            "target_categories": torch.as_tensor(target_cats),
            "dead_ends": dead,
            "cells_edited": n_edits,
            "avg_corrections_per_sample": n_edits / max(B, 1),
            "boards_needing_erasure": int(sum(1 for s in snipped if s)),
            "mean_cells_snipped": float(np.mean(snipped)) if snipped else 0.0,
            "mean_arrival": float(np.mean([lg["arrival"] for lg in logs])),
            "lifetime_boards": self.lifetime_boards,
            "lifetime_dead_ends": self.lifetime_dead_ends,
        }

        if self.certify:
            from compute_cost_ratio import _is_valid_src_sink_path

            from steps_walk import edge_endpoints_array
            ep = edge_endpoints_array(self.grid_size)
            sink = (self.grid_size - 1) * self.grid_size + (self.grid_size - 1)
            bad = [b for b in range(B)
                   if not _is_valid_src_sink_path(
                       edge_assign[b].astype(bool), ep, 0, sink)]
            self.lifetime_uncertified += len(bad)
            info["uncertified_boards"] = bad
            info["certified_boards"] = B - len(bad)

        return adjusted, info


def one_hot_from_categories(node_labels, var_ranges):
    """(B, n_nodes) category indices -> (B, total_dim) one-hot distributions."""
    node_labels = torch.as_tensor(node_labels, dtype=torch.long)
    B = node_labels.shape[0]
    out = torch.zeros(B, int(sum(var_ranges)))
    off = 0
    for k, vr in enumerate(var_ranges):
        out[torch.arange(B), off + node_labels[:, k]] = 1.0
        off += vr
    return out
