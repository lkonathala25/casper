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
