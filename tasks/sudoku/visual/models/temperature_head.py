"""LLM + frozen solver + perception fusion head.

Replaces the scalar calibration head with a proper fusion that recovers
useful downstream uncertainty from cheap frozen signals already available
at training/eval time.

Why this is needed
------------------
The cached LLM adapter encodes one digit per cell as one-hot-like logits
(LLM_CONF at predicted digit, 0 elsewhere). This destroys alternative-
candidate uncertainty that MinEdit needs to repair conflicting assignments.
The frozen nn_solver produces genuine soft distributions; using it as the
primary signal recovers the historical MinEdit/mask_nn operating regime.

Architecture (use_perception_features=True, feat_dim=30)
---------------------------------------------------------
  log_base = w_sol  * log(solver_probs)
           + w_llm  * log_softmax(llm_logits / T_llm)
           + w_perc * perception_logprobs[:,:,1:10]

  features = [llm_logits(9), solver_probs(9), perc_digits(9),
              agree_flag(1), sol_entropy(1), sol_margin(1)]  → 30 dims
  residual = MLP(LayerNorm(features))
  output   = softmax(log_base + residual, dim=-1)

Architecture (use_perception_features=False, feat_dim=21)
----------------------------------------------------------
  log_base = w_sol * log(solver_probs) + w_llm * log_softmax(llm_logits / T_llm)
  features = [llm_logits(9), solver_probs(9),
              agree_flag(1), sol_entropy(1), sol_margin(1)]  → 21 dims

Safe init
---------
  w_sol=1, w_llm=0, (w_perc=0 if used), T_llm=3, MLP final layer = zeros.
  At epoch 0: output = softmax(log(solver_probs)) = solver_probs.
  This is exactly the old MinEdit/mask_nn operating regime.
  Training then learns to blend in LLM and (optionally) perception signal.

use_perception_features flag
-----------------------------
  Stored on the module; controls both the log_base composition and the MLP
  input dimensionality. Checkpoints trained with/without perception are
  incompatible (different residual_mlp.0 input size) — this is intentional.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8


class TemperatureHead(nn.Module):
    """Fusion calibration head combining LLM, solver, and (optionally) perception signals.

    Args:
        hidden_dim: MLP hidden layer width (default 128).
        use_perception_features: when True, include perception log-probs in the
            log-base mixture and as MLP features. When False, only LLM logits
            and solver probs are used. Stored on the module and honored at
            forward time.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        use_perception_features: bool = True,
    ):
        super().__init__()
        self.use_perception_features = use_perception_features

        # Scalar mixture weights over log-distribution streams.
        # Init: w_sol=1, others=0 → pure solver distribution at epoch 0.
        self.w_sol = nn.Parameter(torch.ones(1))
        self.w_llm = nn.Parameter(torch.zeros(1))
        if use_perception_features:
            self.w_perc = nn.Parameter(torch.zeros(1))

        # LLM temperature: T=3 → p_max≈0.73 for LLM_CONF=10.
        self.log_T_llm = nn.Parameter(torch.tensor([math.log(3.0)]))

        # Per-cell feature dimension depends on perception flag:
        #   with:    llm(9) + sol(9) + perc(9) + agree(1) + ent(1) + margin(1) = 30
        #   without: llm(9) + sol(9)            + agree(1) + ent(1) + margin(1) = 21
        _feat_dim = 30 if use_perception_features else 21
        self._feat_dim = _feat_dim
        self.residual_mlp = nn.Sequential(
            nn.LayerNorm(_feat_dim),
            nn.Linear(_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 9),
        )
        # Zero-init final layer so residual=0 at epoch 0.
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

    def forward(
        self,
        llm_logits: torch.Tensor,
        solver_probs: torch.Tensor,
        perception_logprobs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            llm_logits:          (B, 81, 9) one-hot-like LLM adapter logits.
            solver_probs:        (B, 81, 9) frozen nn_solver softmax probs.
            perception_logprobs: (B, 81, 10) perception log-probs, class 0 = blank.
                                 Only used when use_perception_features=True.
        Returns:
            (B, 81, 9) calibrated probability tensor for MinEdit.
        """
        B, C, D = llm_logits.shape  # C=81, D=9

        T_llm = self.log_T_llm.exp().clamp(min=1e-3)
        log_llm = F.log_softmax(llm_logits / T_llm, dim=-1)  # (B, 81, 9)
        log_sol = (solver_probs + _EPS).log()                 # (B, 81, 9)

        # Perception: only when flag is True.
        if self.use_perception_features:
            if perception_logprobs is not None:
                log_perc     = perception_logprobs[:, :, 1:10]  # (B, 81, 9)
                perc_probs_d = log_perc.exp()
            else:
                # Flag is on but no tensor provided — zero-pad gracefully.
                log_perc     = log_sol.new_zeros(B, C, D)
                perc_probs_d = log_sol.new_zeros(B, C, D)
        # else: perception variables are never created or used below.

        # Product-of-experts mixture in log-space.
        log_base = self.w_sol * log_sol + self.w_llm * log_llm
        if self.use_perception_features:
            log_base = log_base + self.w_perc * log_perc

        # Cheap per-cell agreement / uncertainty features (always included).
        llm_arg    = llm_logits.argmax(dim=-1, keepdim=True)               # (B,81,1)
        sol_arg    = solver_probs.argmax(dim=-1, keepdim=True)              # (B,81,1)
        agree      = (llm_arg == sol_arg).float()                           # (B,81,1)
        sol_ent    = -(solver_probs * log_sol).sum(dim=-1, keepdim=True)    # (B,81,1)
        sol_top2, _ = solver_probs.topk(2, dim=-1)
        sol_margin = sol_top2[:, :, :1] - sol_top2[:, :, 1:2]              # (B,81,1)

        # Build feature vector — perception slot only when flag is on.
        if self.use_perception_features:
            feat = torch.cat(
                [llm_logits, solver_probs, perc_probs_d,
                 agree, sol_ent, sol_margin],
                dim=-1,
            )  # (B, 81, 30)
        else:
            feat = torch.cat(
                [llm_logits, solver_probs, agree, sol_ent, sol_margin],
                dim=-1,
            )  # (B, 81, 21)

        flat     = feat.reshape(B * C, self._feat_dim)
        residual = self.residual_mlp(flat).reshape(B, C, D)  # (B, 81, 9)

        return F.softmax(log_base + residual, dim=-1)
