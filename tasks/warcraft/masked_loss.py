import sys as _sys
from pathlib import Path as _Path

# Runnable entry point: put the shared compiler and this task's own directory on
# the path. Sibling task directories are deliberately not added -- they share
# module names (data, model, train, minedit).
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT / "casper", _Path(__file__).resolve().parent):
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


def vectorised_positive_head(probs, targets, eps=1e-3):
    """
    Vectorised L2-minimal edit pulling argmax to `targets`, per row.

    Args:
        probs:   (N, C)  softmax probabilities
        targets: (N,)    long, index per row in [0, C)
        eps:     margin

    Returns:
        p_adj:     (N, C)
        l2:        (N,)
        n_active:  (N,)  long
    """
    N, C = probs.shape
    device = probs.device

    p_target = probs.gather(1, targets.unsqueeze(1))                 # (N, 1)

    target_mask = F.one_hot(targets, C).bool()                       # (N, C)
    active_mask = (p_target < probs + eps) & (~target_mask)          # (N, C)

    n_active = active_mask.sum(dim=1)                                # (N,)
    has_active = n_active > 0                                        # (N,)

    active_sum = (probs * active_mask).sum(dim=1, keepdim=True)      # (N, 1)
    pool = p_target + active_sum                                     # (N, 1)

    denom = (1 + n_active).float().unsqueeze(1)                      # (N, 1)
    new_target_val = (pool + n_active.float().unsqueeze(1) * eps) / denom  # (N, 1)
    new_active_val = (pool - eps) / denom                            # (N, 1)

    # Start from probs, overwrite active positions, then overwrite target positions
    p_adj = torch.where(
        active_mask & has_active.unsqueeze(1),
        new_active_val.expand_as(probs),
        probs,
    )
    target_col = torch.where(has_active.unsqueeze(1), new_target_val, p_target)  # (N, 1)
    p_adj = torch.where(target_mask, target_col.expand_as(probs), p_adj)

    l2 = torch.linalg.vector_norm(p_adj - probs, ord=2, dim=1)
    return p_adj, l2, n_active


class WarcraftMaskedLoss(nn.Module):
    """
    Node-wise analogue of MNIST MaskedLoss. For each node variable:
      Positive term:  -log(p_proj[target])  — projected via MinEdit positive head
      Negative term:  -log(1 - max_wrong(p_raw))  — on raw probs

    Combined: mean over all (batch, node) cells of  loss_pos + lambda_neg * loss_neg.

    Nodes are grouped by category count (2/3/5) and flattened into a single
    (N, C) batch per group, so MinEdit runs fully vectorised per group.
    """

    def __init__(self, var_ranges, lambda_neg=1.0, minedit_eps=1e-3, log_eps=1e-6):
        super().__init__()
        self.var_ranges = var_ranges
        self.lambda_neg = lambda_neg
        self.minedit_eps = minedit_eps
        self.log_eps = log_eps

        # Group node indices by their category count: {n_cats: [(node_idx, offset), ...]}
        groups = defaultdict(list)
        offset = 0
        for n_idx, vr in enumerate(var_ranges):
            groups[vr].append((n_idx, offset))
            offset += vr
        self.groups = dict(groups)

    def forward(self, logits, node_labels):
        """
        Args:
            logits:      (B, total_dim)      pre-softmax, per-node slices concatenated
            node_labels: (B, n_nodes)  long  true category per node

        Returns:
            loss (scalar), info (dict)
        """
        B = logits.shape[0]

        pos_sum = logits.new_zeros(())
        neg_sum = logits.new_zeros(())
        l2_sum = 0.0
        active_sum = 0.0
        bypass_count = 0
        total_cells = 0

        for n_cats, node_list in self.groups.items():
            # Stack node slices along a new "node" dim, then flatten to (B*n_nodes_group, n_cats)
            logit_slices = [logits[:, off:off + n_cats] for _, off in node_list]
            target_slices = [node_labels[:, n_idx] for n_idx, _ in node_list]

            logits_grp = torch.cat(logit_slices, dim=0)              # (B*G, n_cats)
            targets_grp = torch.cat(target_slices, dim=0)            # (B*G,)
            probs_grp = F.softmax(logits_grp, dim=1)                 # (B*G, n_cats)

            p_proj, l2, n_active = vectorised_positive_head(
                probs_grp, targets_grp, eps=self.minedit_eps
            )

            # Positive loss: -log(p_proj[target])
            p_target_proj = p_proj.gather(1, targets_grp.unsqueeze(1)).squeeze(1)
            loss_pos = -torch.log(p_target_proj + self.log_eps)

            # Negative loss: -log(1 - max_wrong(p_raw))
            target_mask = F.one_hot(targets_grp, n_cats).bool()
            p_masked = probs_grp.masked_fill(target_mask, 0.0)
            max_wrong = p_masked.max(dim=1).values
            loss_neg = -torch.log(1 - max_wrong + self.log_eps)

            pos_sum = pos_sum + loss_pos.sum()
            neg_sum = neg_sum + loss_neg.sum()

            l2_sum += l2.sum().item()
            active_sum += n_active.sum().item()
            bypass_count += (n_active == 0).sum().item()
            total_cells += logits_grp.shape[0]

        total_loss = (pos_sum + self.lambda_neg * neg_sum) / total_cells

        info = {
            "loss_pos_mean": (pos_sum / total_cells).item(),
            "loss_neg_mean": (neg_sum / total_cells).item(),
            "mean_l2_edit": l2_sum / total_cells,
            "active_set_size_mean": active_sum / total_cells,
            "bypass_rate": bypass_count / total_cells,
        }
        return total_loss, info


if __name__ == "__main__":
    from node_categories import build_node_categories

    node_info, var_ranges, node_order, total_dim = build_node_categories(12, alphabet='monotone')
    print(f"Nodes: {len(node_order)}, total_dim: {total_dim}")

    B = 4
    torch.manual_seed(0)
    logits = torch.randn(B, total_dim, requires_grad=True)
    node_labels = torch.stack([
        torch.randint(0, vr, (B,)) for vr in var_ranges
    ], dim=1)

    loss_fn = WarcraftMaskedLoss(var_ranges, lambda_neg=1.0)
    loss, info = loss_fn(logits, node_labels)
    loss.backward()

    print(f"Loss: {loss.item():.4f}")
    print(f"Grad norm: {logits.grad.norm().item():.4f}")
    for k, v in info.items():
        print(f"  {k}: {v:.4f}")
