import torch
import torch.nn as nn

from carry_chain import compute_carry_chain  # noqa: F401
from minedit import MinEditL2


class CarryChainConstraintLayer(nn.Module):
    def __init__(self, n_digits, eps=1e-3, projection_backend='closed_form',
                 solver_time_limit=None):
        super().__init__()
        self.n_digits = n_digits
        self.n_output_digits = n_digits + 1
        self.n_images = 2 * n_digits
        self.projection_backend = projection_backend

        if projection_backend == 'closed_form':
            self.minedit = MinEditL2(10, eps=eps)
        elif projection_backend == 'cvxpy':
            from cvxpy_minedit import CVXPYMinEditL2
            self.minedit = CVXPYMinEditL2(
                10, eps=eps, solver_time_limit=solver_time_limit)
        else:
            raise ValueError(f"Unknown projection_backend: {projection_backend!r}")

    def forward(self, probs):
        """Project output-digit distributions onto the carry-chain surface.

        probs layout: [a_0..a_{N-1}, b_0..b_{N-1}, s_0..s_N], 10-wide each.
        """
        B = probs.shape[0]
        N = self.n_digits
        n_images = self.n_images
        n_output = self.n_output_digits

        input_probs = probs[:, :n_images * 10].view(B, n_images, 10)
        output_probs = probs[:, n_images * 10:].view(B, n_output, 10)

        a_preds = input_probs[:, :N].argmax(dim=2)
        b_preds = input_probs[:, N:].argmax(dim=2)

        s_targets = compute_carry_chain(a_preds, b_preds)

        adjusted_output = output_probs.clone()
        l2_per_position = []
        total_constraints = 0

        for j in range(n_output):
            pos_probs = output_probs[:, j]
            current_argmax = pos_probs.argmax(dim=1)
            target = s_targets[:, j]
            needs_fix = (current_argmax != target)

            l2_j = torch.zeros(B, device=probs.device)

            if needs_fix.any():
                total_constraints += needs_fix.sum().item()
                for target_val in target[needs_fix].unique():
                    group_mask = needs_fix & (target == target_val)
                    adj, dist, _na = self.minedit(
                        pos_probs[group_mask],
                        target_category=target_val.item(),
                        is_positive=True,
                    )
                    idx = group_mask.nonzero(as_tuple=False).squeeze(1)
                    expanded = torch.zeros(B, 10, device=probs.device).index_copy(0, idx, adj)
                    adjusted_output[:, j] = torch.where(
                        group_mask.unsqueeze(1), expanded, adjusted_output[:, j]
                    )
                    expanded_l2 = torch.zeros(B, device=probs.device).index_copy(0, idx, dist)
                    l2_j = torch.where(group_mask, expanded_l2, l2_j)

            l2_per_position.append(l2_j)

        l2_stack = torch.stack(l2_per_position, dim=1)
        projection_loss = l2_stack.sum(dim=1).mean()

        adjusted = torch.cat([
            probs[:, :n_images * 10],
            adjusted_output.view(B, -1),
        ], dim=1)

        info = {
            'projection_loss': projection_loss,
            'total_l2_distance': l2_stack.sum().item(),
            'avg_l2_per_sample': l2_stack.sum(dim=1).mean().item(),
            'total_constraints_applied': total_constraints,
        }
        return adjusted, info
