import torch
import torch.nn as nn


class MinEditL2(nn.Module):
    def __init__(self, n_categories, eps=1e-3):
        super().__init__()
        self.n_categories = n_categories
        self.eps = eps

    def _positive_head(self, p, target):
        """Project so argmax(p_adj) == target with margin eps.

        Closed-form KKT: collect categories j != target where p[t] < p[j] + eps
        (the active set), pool their mass with p[t], and redistribute via

            new_target_val = (pool + n_active * eps) / (1 + n_active)
            new_active_val = (pool - eps) / (1 + n_active)

        Categories outside the active set are unchanged. Mass is conserved.
        """
        p_adj = p.clone()

        active = []
        for j in range(self.n_categories):
            if j != target and p[target] < p[j] + self.eps:
                active.append(j)

        if not active:
            return p_adj, torch.tensor(0.0, device=p.device)

        n_active = len(active)
        active_sum = sum(p[j] for j in active)
        pool = p[target] + active_sum

        p_adj[target] = (pool + n_active * self.eps) / (1 + n_active)
        for j in active:
            p_adj[j] = (pool - self.eps) / (1 + n_active)

        l2 = torch.linalg.vector_norm(p_adj - p, ord=2)
        return p_adj, l2

    def _negative_head(self, p, forbidden):
        """Project so argmax(p_adj) != forbidden by promoting the runner-up.

        Samples whose argmax is already != forbidden are bypassed (zero edit).
        """
        current_argmax = p.argmax().item()

        if current_argmax != forbidden:
            return p.clone(), torch.tensor(0.0, device=p.device)

        p_masked = p.clone()
        p_masked[forbidden] = -float('inf')
        runner_up = p_masked.argmax().item()

        return self._positive_head(p, target=runner_up)

    def forward(self, probs, target_category, is_positive):
        adjusted_list = []
        distances = []

        for i in range(probs.shape[0]):
            if is_positive:
                adj, dist = self._positive_head(probs[i], target_category)
            else:
                adj, dist = self._negative_head(probs[i], target_category)
            adjusted_list.append(adj)
            distances.append(dist)

        return torch.stack(adjusted_list), torch.stack(distances)
