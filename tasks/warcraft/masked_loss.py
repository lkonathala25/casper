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
import torch.nn.functional as F


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
