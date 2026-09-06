import torch
import torch.nn as nn

from minedit import MinEditL2


class ConstraintLayer(nn.Module):
    def __init__(self, variable_ranges, constraint_fn=None,
                 projection_backend='closed_form', solver_time_limit=None,
                 eps=1e-3):
        super().__init__()
        self.variable_ranges = variable_ranges
        self.start_indices = self._compute_start_indices(variable_ranges)
        self.n_digits = len(variable_ranges) - 1
        self.constraint_fn = constraint_fn if constraint_fn is not None \
            else (lambda preds: sum(preds))

        self.projection_backend = projection_backend
        n_sum_categories = variable_ranges[-1]
        if projection_backend == 'closed_form':
            self.minedit_sum = MinEditL2(n_sum_categories, eps=eps)
        elif projection_backend == 'cvxpy':
            from cvxpy_minedit import CVXPYMinEditL2
            self.minedit_sum = CVXPYMinEditL2(
                n_sum_categories, eps=eps, solver_time_limit=solver_time_limit)
        else:
            raise ValueError(f"Unknown projection_backend: {projection_backend!r}")

    def _compute_start_indices(self, var_ranges):
        start_indices = [0]
        for size in var_ranges[:-1]:
            start_indices.append(start_indices[-1] + size)
        return start_indices

    def forward(self, probs):
        """Project sum-variable distribution onto the constraint surface."""
        B = probs.shape[0]

        digit_preds = []
        for i in range(self.n_digits):
            start = self.start_indices[i]
            end = start + self.variable_ranges[i]
            digit_preds.append(probs[:, start:end].argmax(dim=1))

        expected_sum = self.constraint_fn(digit_preds)

        sum_start = self.start_indices[-1]
        sum_end = sum_start + self.variable_ranges[-1]
        sum_probs = probs[:, sum_start:sum_end]
        current_sum_pred = sum_probs.argmax(dim=1)

        needs_fix = (current_sum_pred != expected_sum)
        total_l2 = 0.0

        adjusted_sum = sum_probs
        if needs_fix.any():
            for target_val in expected_sum[needs_fix].unique():
                group_mask = needs_fix & (expected_sum == target_val)
                adj, dist, _na = self.minedit_sum(
                    sum_probs[group_mask],
                    target_category=target_val.item(),
                    is_positive=True,
                )
                total_l2 += dist.sum().item()
                S = self.variable_ranges[-1]
                idx = group_mask.nonzero(as_tuple=False).squeeze(1)
                expanded = torch.zeros(B, S, device=probs.device)
                expanded = expanded.index_copy(0, idx, adj)
                adjusted_sum = torch.where(
                    group_mask.unsqueeze(1), expanded, adjusted_sum)

        adjusted = torch.cat([probs[:, :sum_start], adjusted_sum], dim=1)

        info = {
            'total_l2_distance': total_l2,
            'avg_l2_per_sample': total_l2 / B,
            'total_constraints_applied': needs_fix.sum().item(),
        }
        return adjusted, info
