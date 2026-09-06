import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
# Appended, not prepended — see the note in lazy_exact_layer.py about the
# module-name collisions between src/ and src_warcraft/.
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from node_categories import build_node_categories, edge_labels_to_node_labels


def default_artifact(grid_size):
    """Path to the Family 1 degree-constraint artifact for a grid size."""
    return ROOT / "examples" / "warcraft_artifacts" / f"wc_deg_{grid_size}_geometric.txt"


def build_endpoint_matrices(node_info, node_order, total_dim, num_edges):
    """Per-endpoint category->edge incidence matrices.

    Returns (M_u, M_v, counts) where M_u[c, e] = 1 iff category slot c belongs to
    the lower-indexed endpoint of edge e and switches e on; M_v is the same for
    the higher-indexed endpoint. `counts[e]` is how many endpoints contribute,
    which is always 2 on a grid but is kept explicit so the averaging is honest.
    """
    M_u = torch.zeros(total_dim, num_edges)
    M_v = torch.zeros(total_dim, num_edges)

    offsets, off = {}, 0
    for node in node_order:
        offsets[node] = off
        off += node_info[node]['n_cats']

    rank = {node: idx for idx, node in enumerate(node_order)}
    edge_endpoints = {}
    for node in node_order:
        for e in node_info[node]['all_edges']:
            edge_endpoints.setdefault(e, []).append(node)

    for e, nodes in edge_endpoints.items():
        lo, hi = sorted(nodes, key=lambda nd: rank[nd])
        for target, node in ((M_u, lo), (M_v, hi)):
            for cat_idx, pattern in enumerate(node_info[node]['edge_patterns']):
                if pattern.get(e, 0) == 1:
                    target[offsets[node] + cat_idx, e] = 1.0

    counts = torch.tensor([len(edge_endpoints[e]) for e in range(num_edges)],
                          dtype=torch.float32)
    return M_u, M_v, counts


class CategoricalEdgeBridge:
    """Maps per-node categorical distributions to per-edge scores and back.

    Forward (categorical -> edge): each endpoint of an edge induces a marginal
    probability that the edge is on; the score is their mean.

    Backward (edge -> categorical): a degree-legal edge assignment determines
    each node's incident pattern, hence exactly one category under the 'full'
    alphabet. Raises if the assignment is not representable.
    """

    def __init__(self, grid_size=12, alphabet='full'):
        self.grid_size = grid_size
        self.alphabet = alphabet
        self.node_info, self.var_ranges, self.node_order, self.total_dim = \
            build_node_categories(grid_size, alphabet=alphabet)
        self.num_edges = 2 * grid_size * (grid_size - 1)
        self.M_u, self.M_v, self.counts = build_endpoint_matrices(
            self.node_info, self.node_order, self.total_dim, self.num_edges)

    def to(self, device):
        self.M_u = self.M_u.to(device)
        self.M_v = self.M_v.to(device)
        self.counts = self.counts.to(device)
        return self

    def edge_scores(self, probs):
        """(B, total_dim) categorical probabilities -> (B, num_edges) in [0, 1]."""
        return (probs @ self.M_u + probs @ self.M_v) / self.counts

    def endpoint_marginals(self, probs):
        """The two per-endpoint marginals separately, for consistency diagnostics."""
        return probs @ self.M_u, probs @ self.M_v

    def edges_to_categories(self, edge_assign, instance_ids=None):
        """(B, num_edges) binary -> (B, n_nodes) category indices."""
        arr = edge_assign.detach().cpu().numpy() if torch.is_tensor(edge_assign) \
            else edge_assign
        return edge_labels_to_node_labels(
            arr, self.node_info, self.node_order, instance_ids=instance_ids)

    def categories_to_edges(self, node_labels):
        """(B, n_nodes) category indices -> (B, num_edges) binary.

        Union over both endpoints, matching node_preds_to_edge_preds. For a
        globally consistent labeling the two endpoints agree and this is exact.
        """
        if not torch.is_tensor(node_labels):
            node_labels = torch.as_tensor(node_labels)
        B = node_labels.shape[0]
        onehot = torch.zeros(B, self.total_dim)
        off = 0
        for k, vr in enumerate(self.var_ranges):
            onehot[torch.arange(B), off + node_labels[:, k]] = 1.0
            off += vr
        both = (onehot @ self.M_u.cpu()) + (onehot @ self.M_v.cpu())
        return (both > 0).to(torch.int64)


def embed_monotone_head_into_full(state_dict, grid_size, neg_logit=-30.0,
                                  fc_weight_key="backbone.fc.weight",
                                  fc_bias_key="backbone.fc.bias"):
    """Widen a monotone-alphabet output head to the full alphabet.

    The full alphabet is a strict prefix extension of the monotone one at every
    node, so each node's existing category block copies across unchanged and the
    added backward-corner categories get a constant `neg_logit` bias with zero
    weights. Their softmax mass is ~0: the network as trained never prefers a
    backward corner, and only the projection can select one.
    """
    _, mono_vr, _, mono_dim = build_node_categories(grid_size, alphabet='monotone')
    _, full_vr, _, full_dim = build_node_categories(grid_size, alphabet='full')

    W = state_dict[fc_weight_key]
    b = state_dict[fc_bias_key]
    if W.shape[0] != mono_dim:
        raise ValueError(
            f"{fc_weight_key} has {W.shape[0]} rows, expected the monotone head "
            f"size {mono_dim} for a {grid_size}x{grid_size} grid")

    newW = W.new_zeros(full_dim, W.shape[1])
    newb = b.new_full((full_dim,), float(neg_logit))

    src = dst = 0
    for mvr, fvr in zip(mono_vr, full_vr):
        newW[dst:dst + mvr] = W[src:src + mvr]
        newb[dst:dst + mvr] = b[src:src + mvr]
        src += mvr
        dst += fvr

    out = dict(state_dict)
    out[fc_weight_key] = newW
    out[fc_bias_key] = newb
    return out
