from __future__ import annotations

import torch
import torch.nn.functional as F

CASPER_MODES = ("none", "infer_only", "train_only", "train_infer")


def layer_in_training(mode: str) -> bool:
    """Whether --casper-mode runs the layer in the training forward pass (task loss on its output)."""
    if mode not in CASPER_MODES:
        raise ValueError(f"casper mode must be one of {CASPER_MODES}, got {mode!r}")
    return mode in ("train_only", "train_infer")


def layer_at_inference(mode: str) -> bool:
    """Whether --casper-mode applies the layer to the predictions that are evaluated."""
    if mode not in CASPER_MODES:
        raise ValueError(f"casper mode must be one of {CASPER_MODES}, got {mode!r}")
    return mode in ("infer_only", "train_infer")


def resolve_l2_penalty(mode: str, l2_penalty: float | None) -> float:
    """The l2-penalty weight for a run, refusing settings that make an ablation ambiguous.

    A mode that trains through the layer needs the weight set explicitly (1.0 for
    full CASPER, 0 for the w/o-l2-penalty ablation). A mode that does not run the
    layer in training has no penalty to weight, so a non-zero value is an error.
    """
    if layer_in_training(mode):
        if l2_penalty is None:
            raise ValueError(f"--casper-mode {mode} trains through the layer: set --l2-penalty "
                             f"explicitly (1.0 for full CASPER, 0 for the w/o-l2-penalty ablation)")
        if l2_penalty < 0:
            raise ValueError(f"--l2-penalty must be >= 0, got {l2_penalty}")
        return float(l2_penalty)
    if l2_penalty:
        raise ValueError(f"--l2-penalty {l2_penalty} has no effect: --casper-mode {mode} "
                         f"does not run the layer in training")
    return 0.0


class VariableLayout:
    """The categorical variables of a flat (B, sum(var_ranges)) tensor, grouped by domain size.

    Variables with the same domain are gathered into one (B, G, h) block so the
    per-variable losses run as a handful of batched ops whatever the variable count.
    """

    def __init__(self, var_ranges):
        self.var_ranges = list(var_ranges)
        offsets, groups = [], {}
        off = 0
        for v, h in enumerate(self.var_ranges):
            offsets.append(off)
            groups.setdefault(h, []).append(v)
            off += h
        self.total_dim = off
        self._groups = [(h, vars_, [[offsets[v] + j for j in range(h)] for v in vars_])
                        for h, vars_ in sorted(groups.items())]
        self._cache = {}

    def blocks(self, x: torch.Tensor):
        """Yield (variable indices, (B, G, h) block of x) per domain size."""
        if x.shape[1] != self.total_dim:
            raise ValueError(f"expected {self.total_dim} columns, got {x.shape[1]}")
        if x.device not in self._cache:
            self._cache[x.device] = [
                (torch.tensor(vars_, dtype=torch.long, device=x.device),
                 torch.tensor(cols, dtype=torch.long, device=x.device))
                for _h, vars_, cols in self._groups]
        for vars_, cols in self._cache[x.device]:
            yield vars_, x[:, cols]


def _per_variable_sum(nll_blocks, weights):
    """Sum (B, G) per-variable losses over variables (weighted if given), then mean over the batch."""
    total = 0.0
    for vars_, nll in nll_blocks:
        if weights is not None:
            nll = nll * weights[:, vars_]
        total = total + nll.sum(dim=1)
    return total.mean()


def raw_nll(layout: VariableLayout, logits, targets, weights=None):
    """Task loss without the layer: cross-entropy of each variable's raw logits.

    Summed over variables per example and averaged over the batch. targets is
    (B, n_vars) long; weights, if given, is (B, n_vars) per-variable weights.
    """
    def blocks():
        for vars_, block in layout.blocks(logits):
            logp = F.log_softmax(block, dim=2)
            yield vars_, -logp.gather(2, targets[:, vars_].unsqueeze(2)).squeeze(2)
    return _per_variable_sum(blocks(), weights)


def projected_nll(layout: VariableLayout, projected, targets, weights=None, min_prob=1e-8):
    """Task loss through the layer: -log of each variable's projected probability of its target.

    Same reduction as raw_nll. The projected output is a distribution, not logits,
    and the min-edit can leave exact zeros, hence the clamp at min_prob.
    """
    def blocks():
        for vars_, block in layout.blocks(projected):
            p = block.gather(2, targets[:, vars_].unsqueeze(2)).squeeze(2)
            yield vars_, -torch.log(p.clamp_min(min_prob))
    return _per_variable_sum(blocks(), weights)


def l2_penalty_term(projected, probs):
    """½‖z̃ − ŷ‖² over all variables per example, averaged over the batch (Proposition 2)."""
    return 0.5 * (projected - probs).pow(2).sum(dim=1).mean()


def casper_loss(layout: VariableLayout, logits, probs, projected, targets, l2_penalty,
                weights=None):
    """The training objective every task uses: task loss + l2_penalty · ½‖z̃ − ŷ‖².

    With projected=None (the layer is not run in training) the task loss is
    raw_nll on the logits and there is no penalty. Otherwise it is projected_nll
    on the layer's output, with gradients flowing through the layer, plus the
    weighted penalty. Returns (loss, parts): parts holds the unweighted task loss,
    the unweighted penalty term and the mean per-example L2 edit ‖z̃ − ŷ‖.
    """
    if projected is None:
        task = raw_nll(layout, logits, targets, weights)
        return task, {"task_loss": task.item(), "l2_penalty_term": float("nan"),
                      "l2_edit": float("nan")}
    task = projected_nll(layout, projected, targets, weights)
    penalty = l2_penalty_term(projected, probs)
    loss = task + l2_penalty * penalty if l2_penalty else task
    edit = torch.linalg.vector_norm((projected - probs).detach(), dim=1).mean()
    return loss, {"task_loss": task.item(), "l2_penalty_term": penalty.item(),
                  "l2_edit": edit.item()}
