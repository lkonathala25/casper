import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitEncoder(nn.Module):
    """LeNet-5 backbone: (B, 1, 28, 28) -> (B, 10) logits."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        return self.classifier(x)


class SumClassifier(nn.Module):
    """MLP head over concatenated digit logits -> (9*N+1)-way sum logits."""

    def __init__(self, input_dim, n_sum_categories):
        super().__init__()
        hidden = max(256, min(2048, input_dim))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_sum_categories),
        )

    def forward(self, x):
        return self.mlp(x)


class OutputDigitClassifier(nn.Module):
    """MLP head over concatenated digit logits -> (N+1) x 10 output digit logits."""

    def __init__(self, input_dim, n_output_digits):
        super().__init__()
        self.n_output_digits = n_output_digits
        output_dim = n_output_digits * 10
        hidden = max(256, min(2048, input_dim))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class MNISTConstraintNet(nn.Module):
    """End-to-end model for MNIST-Sum or MNIST-Add with constraint projection."""

    def __init__(self, task, variable_ranges, n_inputs, constraint_fn=None,
                 projection_backend='closed_form', solver_time_limit=None):
        super().__init__()
        if task not in ('sum', 'add'):
            raise ValueError(f"task must be 'sum' or 'add', got {task!r}")
        self.task = task
        self.var_ranges = variable_ranges
        self.start_indices = self._compute_start_indices(variable_ranges)
        self.n_inputs = n_inputs

        self.digit_encoder = DigitEncoder()

        if task == 'sum':
            from constraint_layer import ConstraintLayer

            n_sum_categories = variable_ranges[-1]
            self.head = SumClassifier(n_inputs * 10, n_sum_categories)
            self.constraint_layer = ConstraintLayer(
                variable_ranges,
                constraint_fn=constraint_fn,
                projection_backend=projection_backend,
                solver_time_limit=solver_time_limit,
            )
        else:
            from carry_chain_constraint_layer import CarryChainConstraintLayer

            n_digits_per_num = n_inputs // 2
            n_output_digits = n_digits_per_num + 1
            self.head = OutputDigitClassifier(n_inputs * 10, n_output_digits)
            self.constraint_layer = CarryChainConstraintLayer(
                n_digits_per_num,
                projection_backend=projection_backend,
                solver_time_limit=solver_time_limit,
            )

    @staticmethod
    def _compute_start_indices(var_ranges):
        starts = [0]
        for size in var_ranges[:-1]:
            starts.append(starts[-1] + size)
        return starts

    def get_digit_logits(self, x):
        """x: (B, n_inputs, 28, 28) -> list of n_inputs (B, 10) logits."""
        out = []
        for i in range(self.n_inputs):
            out.append(self.digit_encoder(x[:, i:i + 1, :, :]))
        return out

    def get_logits(self, x):
        digit_logits = self.get_digit_logits(x)
        head_logits = self.head(torch.cat(digit_logits, dim=1))
        return torch.cat(digit_logits + [head_logits], dim=1)

    def _logits_to_probs(self, logits):
        out = []
        offset = 0
        for vr in self.var_ranges:
            out.append(F.softmax(logits[:, offset:offset + vr], dim=1))
            offset += vr
        return torch.cat(out, dim=1)

    def forward_diagnostic(self, x):
        """Return raw and projected per-variable probabilities."""
        probs = self._logits_to_probs(self.get_logits(x))
        projected, info = self.constraint_layer(probs.clone())
        return probs, projected, info
