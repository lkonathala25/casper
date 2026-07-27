import torch
import torch.nn as nn
import torch.nn.functional as F

from minedit import MinEditL2


class PlainSumLoss(nn.Module):
    """Unconstrained MNIST-Sum loss: per-input-digit CE plus CE on the sum head.

    Used when the constraint is disabled at training time (constraint modes
    'infer_only' and 'none'). No MinEdit projection is involved, so the sum
    head is supervised directly against the true sum label.
    """

    def forward(self, *args):
        """Args: logits_v1, ..., logits_vN, logits_sum, targets_v1, ..., targets_sum."""
        n_vars = len(args) // 2
        logits_list = list(args[:n_vars])
        targets_list = list(args[n_vars:])
        losses = [F.cross_entropy(logits_list[i], targets_list[i])
                  for i in range(n_vars)]
        total_loss = sum(losses)
        info = {
            'input_losses': [l.item() for l in losses[:-1]],
            'loss_sum': losses[-1].item(),
            'loss_sum_pos': float('nan'),
            'loss_sum_neg': float('nan'),
            'mean_l2_edit': 0.0,
        }
        return total_loss, info


class MaskedLoss(nn.Module):
    """Decomposed MNIST-Sum loss.

    Per-digit cross-entropy plus a two-term sum loss: L_pos pushes the
    correct class up through the MinEdit projection, L_neg pushes the
    largest wrong-class probability down on the raw distribution. Splitting
    them avoids gradient inversion through the projection coupling.
    """

    def __init__(self, n_sum_categories, lambda_neg=1.0, eps=1e-10,
                 constraint_fn=None):
        super().__init__()
        self.minedit_sum = MinEditL2(n_sum_categories)
        self.lambda_neg = lambda_neg
        self.eps = eps
        self.constraint_fn = constraint_fn if constraint_fn is not None \
            else (lambda targets: sum(targets))

    def forward(self, *args):
        """Args: logits_v1, ..., logits_vN, logits_sum, targets_v1, ..., targets_sum."""
        n_args = len(args)
        n_vars = n_args // 2
        logits_list = list(args[:n_vars])
        targets_list = list(args[n_vars:])

        n_input_vars = n_vars - 1
        input_losses = [
            F.cross_entropy(logits_list[i], targets_list[i])
            for i in range(n_input_vars)
        ]

        logits_sum = logits_list[-1]
        targets_sum = targets_list[-1]
        batch_size = logits_sum.shape[0]
        p_sum = F.softmax(logits_sum, dim=1)

        target_sum_values = self.constraint_fn(targets_list[:-1])

        p_sum_proj = p_sum.clone()
        l2_distances = torch.zeros(batch_size, device=p_sum.device)

        current_argmax = p_sum.argmax(dim=1)
        already_satisfied = (current_argmax == target_sum_values)

        if not already_satisfied.all():
            violated = ~already_satisfied
            violated_targets = target_sum_values[violated]
            violated_probs = p_sum[violated]

            proj_violated = torch.zeros_like(violated_probs)
            dist_violated = torch.zeros(violated.sum(), device=p_sum.device)

            for ts in violated_targets.unique():
                ts_mask = (violated_targets == ts)
                proj_violated[ts_mask], dist_violated[ts_mask], _ = \
                    self.minedit_sum(
                        violated_probs[ts_mask],
                        target_category=ts.item(),
                        is_positive=True,
                    )

            p_sum_proj[violated] = proj_violated
            l2_distances[violated] = dist_violated

        p_target_proj = p_sum_proj.gather(1, targets_sum.unsqueeze(1)).squeeze(1)
        loss_pos = -torch.log(p_target_proj + self.eps).mean()

        wrong_mask = torch.ones_like(p_sum, dtype=torch.bool)
        wrong_mask.scatter_(1, targets_sum.unsqueeze(1), False)
        p_sum_masked = p_sum.clone()
        p_sum_masked[~wrong_mask] = 0.0
        max_wrong_prob = p_sum_masked.max(dim=1).values
        loss_neg = -torch.log(1 - max_wrong_prob + self.eps).mean()

        loss_sum = loss_pos + self.lambda_neg * loss_neg
        total_loss = sum(input_losses) + loss_sum

        info = {
            'input_losses': [l.item() for l in input_losses],
            'loss_sum': loss_sum.item(),
            'loss_sum_pos': loss_pos.item(),
            'loss_sum_neg': loss_neg.item(),
            'mean_l2_edit': l2_distances.mean().item(),
        }
        return total_loss, info
