import torch
import torch.nn as nn

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover — surface the error only if forward_group is called
    linear_sum_assignment = None


class VectorisedMinEditL2(nn.Module):
    """
    Batched/vectorised version of MinEditL2.

    Two projection modes are exposed:

    - Pairwise (``forward(..., pairwise=True)``, the historical path):
      enforces a single ``argmax == t`` or ``argmax != t`` on one variable,
      exact closed-form KKT. This is what every existing caller uses and what
      the pairwise SudokuConstraintLayer loop drives.

    - Joint k-ary (``forward_group``): enforces AllDifferent across K cells
      simultaneously by solving a linear assignment (Hungarian) on the
      per-cell negative-log-prob matrix to pick the MAP permutation, then
      running the existing positive head on each cell with its assigned
      digit as target. Produces a per-group permutation in a single call.

    The forward signature is preserved: ``pairwise`` defaults to True so
    existing callers and unit tests continue to route through the pairwise
    projection unchanged.
    """

    def __init__(self, n_categories, eps=1e-3):
        super().__init__()
        self.n_categories = n_categories
        self.eps = eps

    def _positive_head_batch(self, p, target):
        """
        Enforce argmax = target for every sample in the batch.

        Args:
            p: (B, C) probability vectors
            target: int — single target category (same for all samples)

        Returns:
            p_adj: (B, C) adjusted probabilities
            l2:    (B,)   per-sample L2 distances
        """
        B, C = p.shape

        # (B, 1) probability at the target category
        p_target = p[:, target : target + 1]

        # Active set: categories j != target where p_target < p_j + eps
        # Build target-exclusion mask without in-place op
        target_exclude = torch.ones(C, dtype=torch.bool, device=p.device)
        target_exclude[target] = False
        active_mask = (p_target < p + self.eps) & target_exclude.unsqueeze(0)  # (B, C)

        n_active = active_mask.sum(dim=1)              # (B,)
        has_active = n_active > 0                      # (B,)

        # Pool = p[target] + sum of active probabilities
        active_sum = (p * active_mask).sum(dim=1)      # (B,)
        pool = p[:, target] + active_sum               # (B,)

        # KKT closed-form (same formula as the scalar version)
        denom = (1 + n_active).float()                 # (B,)
        new_target_val = (pool + n_active.float() * self.eps) / denom   # (B,)
        new_active_val = (pool - self.eps) / denom                      # (B,)

        # Build p_adj out-of-place using torch.where
        # Start from p, apply active replacement, then apply target replacement
        active_replacement = new_active_val.unsqueeze(1).expand_as(p)   # (B, C)
        update_mask = active_mask & has_active.unsqueeze(1)             # (B, C)
        p_adj = torch.where(update_mask, active_replacement, p)

        # Build target column: replace only where has_active
        target_col = torch.where(has_active, new_target_val, p[:, target])  # (B,)
        # Scatter target column into p_adj out-of-place
        target_mask = torch.zeros(C, dtype=torch.bool, device=p.device)
        target_mask[target] = True
        target_mask = target_mask.unsqueeze(0).expand_as(p)             # (B, C)
        p_adj = torch.where(target_mask, target_col.unsqueeze(1).expand_as(p), p_adj)

        l2 = torch.linalg.vector_norm(p_adj - p, ord=2, dim=1)         # (B,)
        return p_adj, l2, n_active

    def _negative_head_batch(self, p, forbidden):
        """
        Enforce argmax != forbidden for every sample in the batch.

        Samples whose argmax is already != forbidden are bypassed (zero edit).
        For the rest, the runner-up category is promoted via _positive_head_batch
        with per-sample targets.

        Args:
            p: (B, C) probability vectors
            forbidden: int — forbidden category

        Returns:
            p_adj: (B, C) adjusted probabilities
            l2:    (B,)   per-sample L2 distances
        """
        B, C = p.shape
        current_argmax = p.argmax(dim=1)                        # (B,)
        needs_fix = (current_argmax == forbidden)               # (B,)

        # Fast path: nothing to fix
        if not needs_fix.any():
            return p.clone(), torch.zeros(B, device=p.device), torch.zeros(B, dtype=torch.long, device=p.device)

        # Find per-sample runner-up (highest prob excluding forbidden) — out-of-place
        forbidden_mask = torch.zeros(C, dtype=torch.bool, device=p.device)
        forbidden_mask[forbidden] = True
        p_masked = torch.where(forbidden_mask.unsqueeze(0), torch.tensor(-float('inf'), device=p.device), p)
        runner_up = p_masked.argmax(dim=1)                      # (B,)

        # Apply positive-head projection with per-sample targets — out-of-place
        # Process all needs_fix samples together, then merge back with torch.where
        # First, build a full-batch adjusted tensor by processing each runner-up group
        # and accumulating via torch.where (each sample belongs to exactly one group)
        p_adj = p
        l2 = torch.zeros(B, device=p.device)
        n_active = torch.zeros(B, dtype=torch.long, device=p.device)

        for target_val in runner_up[needs_fix].unique():
            group_mask = needs_fix & (runner_up == target_val)   # (B,) bool
            if not group_mask.any():
                continue
            adj, dist, na = self._positive_head_batch(p[group_mask], target_val.item())
            # Expand adj (G, C) back to (B, C) by placing into a zeros tensor at the right indices
            # and use torch.where to merge
            expanded_adj = torch.zeros_like(p)
            expanded_l2 = torch.zeros(B, device=p.device)
            expanded_na = torch.zeros(B, dtype=torch.long, device=p.device)
            idx = group_mask.nonzero(as_tuple=False).squeeze(1)
            # Use index_copy (out-of-place via new tensor)
            expanded_adj = expanded_adj.index_copy(0, idx, adj)
            expanded_l2 = expanded_l2.index_copy(0, idx, dist)
            expanded_na = expanded_na.index_copy(0, idx, na)
            p_adj = torch.where(group_mask.unsqueeze(1), expanded_adj, p_adj)
            l2 = torch.where(group_mask, expanded_l2, l2)
            n_active = torch.where(group_mask, expanded_na, n_active)

        return p_adj, l2, n_active

    def _positive_head_batch_per_sample_targets(self, p, targets):
        """
        Positive head with per-sample targets — one target index per row of ``p``.

        Wraps ``_positive_head_batch`` by bucketing samples that share the
        same target into sub-batches (same pattern as ``_negative_head_batch``
        does for runner-up routing), then merging the sub-batch results back
        with ``torch.where`` to keep the whole thing differentiable and
        out-of-place.

        Args:
            p:       (B, C) probability vectors
            targets: (B,)   int64 target category per sample

        Returns:
            p_adj:    (B, C) adjusted probabilities
            l2:       (B,)   per-sample L2 distances
            n_active: (B,)   active set sizes per sample
        """
        B, C = p.shape
        p_adj = p
        l2 = torch.zeros(B, device=p.device)
        n_active = torch.zeros(B, dtype=torch.long, device=p.device)

        for target_val in targets.unique():
            group_mask = (targets == target_val)
            if not group_mask.any():
                continue
            adj, dist, na = self._positive_head_batch(p[group_mask], target_val.item())
            expanded_adj = torch.zeros_like(p)
            expanded_l2 = torch.zeros(B, device=p.device)
            expanded_na = torch.zeros(B, dtype=torch.long, device=p.device)
            idx = group_mask.nonzero(as_tuple=False).squeeze(1)
            expanded_adj = expanded_adj.index_copy(0, idx, adj)
            expanded_l2 = expanded_l2.index_copy(0, idx, dist)
            expanded_na = expanded_na.index_copy(0, idx, na)
            p_adj = torch.where(group_mask.unsqueeze(1), expanded_adj, p_adj)
            l2 = torch.where(group_mask, expanded_l2, l2)
            n_active = torch.where(group_mask, expanded_na, n_active)

        return p_adj, l2, n_active

    def forward_group(self, group_probs):
        """
        Joint AllDifferent projection on K cells simultaneously.

        Algorithm (per sample):
          1. Build the KxK cost matrix ``C[k, d] = -log p_k[d]``.
          2. Solve the linear assignment problem (Hungarian) on C to obtain
             the MAP permutation ``sigma`` that minimises sum-cost, i.e.
             maximises the joint log-likelihood of assigning each cell to a
             unique digit.
          3. For each cell position k, apply the pairwise positive head with
             ``target = sigma(k)``. The positive head guarantees
             ``argmax == sigma(k)`` pointwise, so the post-projection group
             is a permutation by construction: 100% AllDifferent on this
             single group in one shot.

        Caveat: applied per-group inside SudokuConstraintLayer, cross-group
        overlap can still break already-processed groups (a row projection
        may change a cell that column/box projections must then revisit).
        Convergence to 100% over all 27 groups is an empirical property of
        the iterated block-coordinate loop, not a per-shot guarantee.

        Args:
            group_probs: (B, K, C) — K cells, C-dim distribution each.
                         Requires K == C (i.e. K digits across K cells).

        Returns:
            adjusted: (B, K, C)
            l2:       (B,)   summed L2 edit across the K cells per sample
            n_active: (B,)   summed active-set sizes across the K cells
        """
        if linear_sum_assignment is None:
            raise RuntimeError(
                "scipy.optimize.linear_sum_assignment is required for the joint "
                "AllDifferent projection. Install scipy or run with --pairwise."
            )
        B, K, C = group_probs.shape
        assert K == C, f"Joint projection requires K == C, got K={K}, C={C}"

        # Hungarian on CPU-numpy — K=9 is tiny, overhead is negligible.
        # .detach() is fine: the assignment selector is piecewise-constant;
        # gradients still flow through the per-cell positive head below.
        cost = -(group_probs.detach().clamp(min=1e-12)).log()              # (B, K, K)
        cost_np = cost.cpu().numpy()

        assignments = torch.empty(B, K, dtype=torch.long, device=group_probs.device)
        for b in range(B):
            _row_ind, col_ind = linear_sum_assignment(cost_np[b])
            # linear_sum_assignment returns row_ind sorted 0..K-1, so col_ind
            # directly indexes the assigned digit per cell position.
            assignments[b] = torch.as_tensor(
                col_ind, dtype=torch.long, device=group_probs.device,
            )

        adj_parts = []
        l2_total = torch.zeros(B, device=group_probs.device)
        na_total = torch.zeros(B, dtype=torch.long, device=group_probs.device)
        for k in range(K):
            targets_k = assignments[:, k]                                  # (B,)
            p_k = group_probs[:, k, :]                                     # (B, C)
            p_k_new, l2_k, na_k = self._positive_head_batch_per_sample_targets(
                p_k, targets_k,
            )
            adj_parts.append(p_k_new)
            l2_total = l2_total + l2_k
            na_total = na_total + na_k
        adjusted = torch.stack(adj_parts, dim=1)                           # (B, K, C)
        return adjusted, l2_total, na_total

    def forward(self, probs, target_category=None, is_positive=None, pairwise=True):
        """
        Pairwise mode (``pairwise=True``, default — backward compatible):
            probs:           (B, C) softmax probabilities
            target_category: int
            is_positive:     bool (True=enforce, False=forbid)
            -> (adjusted (B, C), l2 (B,), n_active (B,))

        Joint mode (``pairwise=False``):
            probs:           (B, K, C) group tensor; requires K == C
            target_category, is_positive: ignored
            -> (adjusted (B, K, C), l2 (B,), n_active (B,))
        """
        if pairwise:
            if is_positive:
                return self._positive_head_batch(probs, target_category)
            else:
                return self._negative_head_batch(probs, target_category)
        return self.forward_group(probs)


if __name__ == "__main__":
    # Minimal self-test: confirm the module runs, produces valid probability
    # simplex outputs, and respects the argmax constraint.
    torch.manual_seed(0)
    C = 9
    B = 32
    m = VectorisedMinEditL2(n_categories=C)

    p = torch.softmax(torch.randn(B, C), dim=1)

    # Positive head
    adj, l2, na = m(p, target_category=4, is_positive=True)
    assert (adj.argmax(dim=1) == 4).all(), "Positive head failed"
    assert torch.allclose(adj.sum(dim=1), torch.ones(B), atol=1e-4), "Not on simplex"
    print(f"Positive head OK — mean L2 edit: {l2.mean().item():.4f}")

    # Negative head
    adj, l2, na = m(p, target_category=4, is_positive=False)
    assert (adj.argmax(dim=1) != 4).all(), "Negative head failed"
    assert torch.allclose(adj.sum(dim=1), torch.ones(B), atol=1e-4), "Not on simplex"
    print(f"Negative head OK — mean L2 edit: {l2.mean().item():.4f}")

    # Joint head — after projection, each sample's group is a permutation.
    K = C
    group = torch.softmax(torch.randn(B, K, C), dim=2)
    adj_g, l2_g, na_g = m.forward_group(group)
    preds = adj_g.argmax(dim=2)                                            # (B, K)
    sorted_preds, _ = preds.sort(dim=1)
    is_perm = (sorted_preds[:, 1:] != sorted_preds[:, :-1]).all(dim=1)
    assert is_perm.all(), "Joint head did not yield per-group permutations"
    assert torch.allclose(adj_g.sum(dim=2), torch.ones(B, K), atol=1e-4), \
        "Joint head output not on simplex"
    print(f"Joint head OK — mean L2 edit (sum over K): {l2_g.mean().item():.4f}")
