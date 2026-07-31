from __future__ import annotations

import torch
import torch.nn as nn

from casper_layer import CasperLayer
from constraints import Constraints
from sudoku_groups import build_groups


class LocalGroupCasper(nn.Module):
    """Per-group CASPER inference over a Sudoku board.

    Runs Algorithm 1 INFER once per ALLDIFFERENT group, in the order given by
    `groups`, against a single compiled clique that is reused for every group
    under a relabelling of local slot -> board cell. Every group is a clique of
    `size` mutually-distinct variables over `size` values, so all 3g groups are
    the same constraint system up to variable renaming and only one compile is
    needed.

    This buys tractability, not Theorem 4. Each group is guaranteed to be
    satisfied at the moment its own pass runs, but passes share the board and
    write in place, so a later group can overwrite a correction an earlier
    group made. Rows, columns and boxes each partition the board, which means
    whichever of the three blocks runs last survives intact and the other two
    carry no guarantee. The output is expected to be locally consistent and
    globally inconsistent; downstream legality checking is still required.
    """

    def __init__(
        self,
        group_constraints: Constraints,
        groups: list[tuple[str, tuple[int, ...]]] | None = None,
        grid_size: int = 9,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.n_cells = grid_size * grid_size
        self.domain = grid_size
        self.groups = groups if groups is not None else build_groups(grid_size)

        domains = list(group_constraints.var_domains)
        if len(domains) != grid_size or any(d != grid_size for d in domains):
            raise ValueError(
                f"compiled group has domains {domains}; expected "
                f"{grid_size} variables of domain {grid_size}"
            )
        for name, cells in self.groups:
            if len(cells) != grid_size:
                raise ValueError(
                    f"group {name} has {len(cells)} cells; expected {grid_size}"
                )

        self.layer = CasperLayer(group_constraints, eps=eps)
        for name, cells in self.groups:
            self.register_buffer(
                f"_idx_{name}", torch.tensor(cells, dtype=torch.long), persistent=False
            )

    def forward(self, probs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Project a batch of boards group by group.

        Args:
            probs: (B, n_cells * domain) per-cell softmax probs, cells in
                   row-major order.

        Returns:
            adjusted: (B, n_cells * domain) — post-INFER probs.
            info: diagnostics; `projection_loss` is differentiable.
        """
        B, total = probs.shape
        expected = self.n_cells * self.domain
        if total != expected:
            raise ValueError(f"expected {expected} columns, got {total}")

        board = probs.view(B, self.n_cells, self.domain)

        projection_loss = probs.new_zeros(())
        total_l2 = 0.0
        total_projected = 0
        per_group: dict[str, int] = {}
        edits_by_pass: list[dict] = []

        for pass_idx, (name, _cells) in enumerate(self.groups):
            idx = getattr(self, f"_idx_{name}")
            before = board.argmax(dim=2)
            sub = board.index_select(1, idx).reshape(B, self.grid_size * self.domain)

            adjusted_sub, sub_info = self.layer(sub)

            board = board.index_copy(
                1, idx, adjusted_sub.view(B, self.grid_size, self.domain)
            )

            changed = (board.argmax(dim=2) != before).nonzero(as_tuple=False)
            if changed.numel():
                edits_by_pass.append({
                    "pass": pass_idx,
                    "group": name,
                    "cells": [(int(b), int(c)) for b, c in changed.tolist()],
                })

            projection_loss = projection_loss + sub_info["projection_loss"]
            total_l2 += sub_info["total_l2_distance"]
            per_group[name] = int(sub_info["total_constraints_applied"])
            total_projected += per_group[name]

        adjusted = board.reshape(B, total)
        info = {
            "projection_loss": projection_loss,
            "total_l2_distance": total_l2,
            "avg_l2_per_sample": total_l2 / B if B else 0.0,
            "total_constraints_applied": total_projected,
            "cells_projected_per_group": per_group,
            "edits_by_pass": edits_by_pass,
            "final_assignments": extract_assignments(
                adjusted, self.n_cells, self.domain
            ),
        }
        return adjusted, info


def alldiff_group_constraints(size: int = 9) -> Constraints:
    """Closed-form ALLDIFFERENT feasibility for one group, no compile needed.

    S_i(a_<i) is just the domain minus the values the group's earlier cells
    already took. Used as an independent oracle against the compiled artifact.
    """

    def feasible(step: int, prior_assignment: dict[int, int], var_domain: int):
        used = set(prior_assignment.values())
        return [v for v in range(var_domain) if v not in used]

    return Constraints.from_function(
        feasible,
        [size] * size,
        ordering=list(range(size)),
        neighbours={v: [u for u in range(size) if u != v] for v in range(size)},
    )


def extract_assignments(
    probs: torch.Tensor, n_cells: int, domain: int
) -> list[list[int]]:
    """Per-sample argmax over each cell's domain slice."""
    B = probs.shape[0]
    grid = probs.view(B, n_cells, domain).argmax(dim=2)
    return [[int(v) for v in row] for row in grid.tolist()]


def group_violations(
    grid: list[int], groups: list[tuple[str, tuple[int, ...]]], domain: int
) -> list[str]:
    """Names of the groups in `grid` that are not a permutation of the domain."""
    target = list(range(domain))
    return [
        name
        for name, cells in groups
        if sorted(grid[c] for c in cells) != target
    ]
