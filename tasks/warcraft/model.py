from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from constraint_layer import WarcraftConstraintLayer
from node_categories import node_preds_to_edge_preds


# =============================================================================
# Model
# =============================================================================

class ResNet18NodeCategorical(nn.Module):
    """ResNet-18 (no pretrained weights) -> per-node softmax over routing categories.

    `constraint_mode` selects the projection: 'monotone' is the original
    down/right flow sweep, 'lazy_exact' is the compiled degree clauses plus ring
    elimination, 'steps' is the sequential min-edit projection over the
    time-unrolled steps encoding. The mode must match the alphabet
    `var_ranges` was built with, which the constructor now enforces.
    """

    def __init__(self, var_ranges, grid_size, processing_order='row',
                 constraint_mode='monotone', artifact=None, alphabet='full',
                 max_rounds=20, certify=False, steps_artifact=None):
        super().__init__()
        self.var_ranges = var_ranges
        self.total_dim = sum(var_ranges)
        self.constraint_mode = constraint_mode

        # Trained from scratch — no pretrained weights.
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, self.total_dim)

        if constraint_mode == 'monotone':
            self.constraint_layer = WarcraftConstraintLayer(
                grid_size=grid_size, processing_order=processing_order
            )
        elif constraint_mode == 'steps':
            import sys
            from pathlib import Path
            steps_dir = Path(__file__).resolve().parent
            if str(steps_dir) not in sys.path:
                sys.path.append(str(steps_dir))
            from steps_layer import StepsProjectionLayer
            self.constraint_layer = StepsProjectionLayer(
                grid_size=grid_size, alphabet=alphabet, certify=certify,
                compiled_artifact=steps_artifact,
            )
        else:
            from lazy_exact_layer import build_constraint_layer
            self.constraint_layer = build_constraint_layer(
                constraint_mode, grid_size=grid_size,
                processing_order=processing_order, artifact=artifact,
                alphabet=alphabet, max_rounds=max_rounds,
            )

        # A layer built for a different alphabet than the head silently
        # mis-slices every node block and returns a correctly-shaped, wrong
        # answer. Make that class of bug impossible to reach.
        layer_dim = getattr(self.constraint_layer, 'total_dim', None)
        if layer_dim != self.total_dim:
            raise ValueError(
                f"constraint layer total_dim {layer_dim} != head dim "
                f"{self.total_dim}: constraint_mode={constraint_mode!r} does "
                f"not match the alphabet the head was built for"
            )

    def forward(self, x, apply_constraints=False):
        logits = self.backbone(x)

        probs = torch.zeros_like(logits)
        idx = 0
        for vr in self.var_ranges:
            probs[:, idx:idx + vr] = F.softmax(logits[:, idx:idx + vr], dim=1)
            idx += vr

        if apply_constraints:
            projected, info = self.constraint_layer(probs.detach())
            return logits, probs, projected, info

        return logits, probs, probs, {}


# =============================================================================
# Loss (base CE)
# =============================================================================

def node_categorical_loss_vectorised(logits, node_labels, var_ranges,
                                     skip_weight=1.0, path_weight=1.0):
    """Weighted per-node CE, batched over nodes that share a category count.

    skip_weight / path_weight defaults of 1.0 reduce to plain uniform CE; raise
    path_weight to up-weight on-path cells at larger grid sizes where the skip
    class dominates.
    """
    groups = defaultdict(list)
    offset = 0
    for n_idx, vr in enumerate(var_ranges):
        groups[vr].append((n_idx, offset))
        offset += vr

    total_loss = logits.new_zeros(())
    total_weight = logits.new_zeros(())

    for vr, node_list in groups.items():
        logits_stack = []
        targets_stack = []
        for n_idx, off in node_list:
            logits_stack.append(logits[:, off:off + vr])
            targets_stack.append(node_labels[:, n_idx])

        logits_cat = torch.cat(logits_stack, dim=0)
        targets_cat = torch.cat(targets_stack, dim=0)

        if vr == 2:
            weights = torch.full_like(targets_cat, path_weight, dtype=logits_cat.dtype)
        else:
            weights = torch.where(
                targets_cat == 0,
                torch.tensor(skip_weight, dtype=logits_cat.dtype, device=logits_cat.device),
                torch.tensor(path_weight, dtype=logits_cat.dtype, device=logits_cat.device),
            )

        loss = F.cross_entropy(logits_cat, targets_cat, reduction='none')
        total_loss = total_loss + (loss * weights).sum()
        total_weight = total_weight + weights.sum()

    return total_loss / total_weight


# =============================================================================
# Metrics
# =============================================================================

def is_valid_path(edge_vec, edges, N, source=(0, 0), target=None):
    """True iff edge_vec encodes a single simple path from source to target."""
    if target is None:
        target = (N - 1, N - 1)
    adj = {(i, j): [] for i in range(N) for j in range(N)}
    active_count = 0
    for idx, (r1, c1, r2, c2) in enumerate(edges):
        if edge_vec[idx] > 0.5:
            adj[(r1, c1)].append((r2, c2))
            adj[(r2, c2)].append((r1, c1))
            active_count += 1
    if active_count == 0:
        return False
    if len(adj[source]) != 1 or len(adj[target]) != 1:
        return False
    for node, neighbours in adj.items():
        if node in (source, target):
            continue
        if len(neighbours) not in (0, 2):
            return False
    visited = set()
    current = source
    prev = None
    for _ in range(active_count + 1):
        if current in visited:
            return False
        visited.add(current)
        if current == target:
            return len(visited) == active_count + 1
        next_node = None
        for n in adj[current]:
            if n != prev:
                next_node = n
                break
        if next_node is None:
            return False
        prev = current
        current = next_node
    return False


def compute_metrics(node_probs, edge_labels, edges, node_info, node_order, num_edges, N):
    """Exact / Hamming / Consistent from node-level probs."""
    edge_preds = node_preds_to_edge_preds(node_probs, node_info, node_order, num_edges)
    labels = edge_labels.float()
    preds = edge_preds.float()

    exact = (preds == labels).all(dim=1).float().mean().item() * 100
    hamming = (preds == labels).float().mean().item() * 100

    consistent = 0
    preds_np = preds.cpu().numpy()
    for i in range(preds_np.shape[0]):
        if is_valid_path(preds_np[i], edges, N):
            consistent += 1
    consistent = consistent / preds_np.shape[0] * 100

    return {"exact": exact, "hamming": hamming, "consistent": consistent}
