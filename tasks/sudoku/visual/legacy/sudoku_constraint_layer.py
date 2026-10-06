import torch
import torch.nn as nn
import torch.nn.functional as F
from legacy.vectorised_minedit import VectorisedMinEditL2
from group_order_utils import validate_group_order


class SudokuConstraintLayer(nn.Module):
    """
    Fast Sudoku-specific constraint layer.

    Input:  (B, 81, 9) softmax probabilities per cell (digits 1..9 at index 0..8).
    Output: (B, 81, 9) projected probabilities satisfying AllDifferent on all
            27 groups (9 rows, 9 cols, 9 boxes) by their argmax.

    Two projection policies, selected at construction by ``pairwise``:

    - ``pairwise=True`` (historical path): for each group, detect duplicate
      argmax digits, keep the highest-confidence cell per duplicate set, push
      the rest off that digit via negative-head MinEdit. Iterate up to
      ``max_iterations`` to resolve cascading conflicts. Local greedy repair
      — plateaus at ~89% AllDifferent sat in practice.

    - ``pairwise=False`` (default, joint k-ary AllDifferent): for each group,
      solve a linear assignment (Hungarian) on the 9x9 negative-log-prob
      matrix to pick the MAP permutation, then project each of the 9 cells
      to its assigned digit via the pairwise positive head. A single pass
      makes the group AllDifferent-consistent by construction; the outer
      iteration then resolves cross-group cascades (every cell is in 3
      overlapping groups — row, column, box).

    ``group_order`` controls the sequence in which the 27 groups are visited
    within each outer iteration (it does not change which 9 cells belong to
    a group, nor the Hungarian assignment logic). It must be a length-27
    permutation of the canonical indices 0..26 from ``_build_groups()``
    (0..8 rows, 9..17 columns, 18..26 boxes); ``None`` (default) keeps the
    canonical 0..26 order, matching historical behaviour exactly. Primarily
    meaningful for the joint ``pairwise=False`` path, where the order groups
    are projected in can affect which cross-group cascades occur first.
    """

    def __init__(self, grid_size=9, eps=1e-3, pairwise=False, group_order=None):
        super().__init__()
        assert grid_size == 9, "Only 9x9 Sudoku supported"
        self.grid_size = grid_size
        self.n_cells = grid_size * grid_size
        self.pairwise = pairwise
        self.minedit = VectorisedMinEditL2(n_categories=grid_size, eps=eps)

        canonical_groups = self._build_groups()
        n_groups = len(canonical_groups)
        # Kept for debug visibility alongside the active order below.
        self.canonical_group_order = list(range(n_groups))
        if group_order is None:
            active_order = list(range(n_groups))
        else:
            active_order = validate_group_order(list(group_order))
        self.group_order = active_order
        groups = [canonical_groups[i] for i in active_order]

        # (27, 9) tensor — each row is the 9 cell indices of one group, in
        # `active_order` processing order (== canonical order by default).
        # Registered as buffer so .to(device) moves it.
        self.register_buffer('groups', torch.tensor(groups, dtype=torch.long))

    @staticmethod
    def _build_groups():
        groups = []
        for r in range(9):
            groups.append([r * 9 + c for c in range(9)])
        for c in range(9):
            groups.append([r * 9 + c for r in range(9)])
        for br in range(3):
            for bc in range(3):
                groups.append([
                    (br * 3 + dr) * 9 + (bc * 3 + dc)
                    for dr in range(3) for dc in range(3)
                ])
        return groups

    def forward(self, cell_probs, max_iterations=10):
        """
        Args:
            cell_probs: (B, 81, 9) softmax per cell.
            max_iterations: max outer passes through all 27 groups.
                Default bumped from 3 to 10 — typical trained-network
                satisfaction reaches ~99%+ within 5-8 iterations; 3 leaves
                ~12% residual violations.

        Returns:
            adjusted: (B, 81, 9) projected probabilities.
            info: dict with l2 / edit / convergence stats.
        """
        B, N, C = cell_probs.shape
        assert N == self.n_cells and C == self.grid_size

        adjusted = cell_probs
        total_l2 = 0.0
        total_edits = 0
        n_iter_done = 0

        for iteration in range(max_iterations):
            n_iter_done = iteration + 1
            any_edit = False

            for g in range(self.groups.shape[0]):
                group_idx = self.groups[g]                       # (9,)

                if self.pairwise:
                    # ------------------------------------------------------
                    # PAIRWISE PATH — per-digit duplicate repair via
                    # negative-head MinEdit. Kept byte-for-byte as the
                    # historical behaviour so existing experiments (suffix
                    # ``_pairwise``) remain reproducible.
                    # ------------------------------------------------------
                    for d in range(self.grid_size):
                        # REFRESH the snapshot inside the digit loop. Earlier
                        # digit-d edits may have moved cells whose new argmax is
                        # the digit we're about to process; a stale outer
                        # snapshot would miss those cascaded conflicts and push
                        # them to the next outer iteration (or silently leave
                        # them unresolved if we hit max_iterations).
                        group_probs = adjusted[:, group_idx, :]      # (B, 9, 9)
                        group_preds = group_probs.argmax(dim=2)      # (B, 9)

                        has_d = (group_preds == d)                   # (B, 9)
                        counts = has_d.sum(dim=1)                    # (B,)
                        has_conflict = counts > 1                    # (B,)

                        if not has_conflict.any():
                            continue

                        # Keeper per sample: cell with max p[d] among has_d cells.
                        conf_at_d = group_probs[:, :, d]             # (B, 9)
                        masked_conf = torch.where(
                            has_d, conf_at_d,
                            torch.full_like(conf_at_d, -1.0),
                        )
                        keeper_local = masked_conf.argmax(dim=1)     # (B,)
                        keeper_mask = F.one_hot(
                            keeper_local, num_classes=self.grid_size
                        ).bool()                                     # (B, 9)

                        # Edit set: predicts d, not keeper, sample in conflict.
                        edit_mask = (
                            has_d
                            & ~keeper_mask
                            & has_conflict.unsqueeze(1)
                        )                                            # (B, 9)

                        if not edit_mask.any():
                            continue

                        sample_idx, local_idx = edit_mask.nonzero(as_tuple=True)
                        global_idx_flat = group_idx[local_idx]       # (N_edit,)

                        # Gather probs for cells to edit (differentiable).
                        probs_to_edit = adjusted[sample_idx, global_idx_flat]

                        adj, l2, _ = self.minedit(
                            probs_to_edit, target_category=d, is_positive=False,
                            pairwise=True,
                        )

                        # Scatter the edits back out-of-place via torch.where.
                        update_tensor = torch.zeros_like(adjusted)
                        update_tensor[sample_idx, global_idx_flat] = adj
                        update_mask_2d = torch.zeros(
                            B, N, dtype=torch.bool, device=adjusted.device
                        )
                        update_mask_2d[sample_idx, global_idx_flat] = True
                        adjusted = torch.where(
                            update_mask_2d.unsqueeze(-1),
                            update_tensor,
                            adjusted,
                        )

                        total_l2 += l2.sum().item()
                        total_edits += sample_idx.numel()
                        any_edit = True
                else:
                    # ------------------------------------------------------
                    # JOINT k-ARY ALLDIFFERENT PATH — one Hungarian
                    # assignment per group plus per-cell positive-head
                    # projection. After this block runs on group g, every
                    # sample that needed a fix has group g as a permutation
                    # (100% AllDifferent on group g by construction).
                    # Cross-group cascades are resolved by the outer
                    # ``for iteration`` loop.
                    # ------------------------------------------------------
                    group_probs = adjusted[:, group_idx, :]          # (B, 9, 9)
                    group_preds = group_probs.argmax(dim=2)          # (B, 9)

                    # Fast-path: groups already a permutation need no edit.
                    sorted_preds, _ = group_preds.sort(dim=1)
                    is_perm = (
                        sorted_preds[:, 1:] != sorted_preds[:, :-1]
                    ).all(dim=1)                                     # (B,)
                    needs_fix = ~is_perm                             # (B,)

                    if not needs_fix.any():
                        continue

                    adj_fix, l2_fix, _ = self.minedit.forward_group(
                        group_probs[needs_fix]
                    )                                                # (Nf, 9, 9)

                    # Scatter the 9 edited cells back into `adjusted` for
                    # samples that were not already a permutation. All K
                    # positions of the group get rewritten for these samples.
                    sample_idx_needs = needs_fix.nonzero(as_tuple=False).squeeze(1)
                    flat_s = sample_idx_needs.view(-1, 1).expand(
                        -1, self.grid_size
                    ).reshape(-1)                                    # (Nf*9,)
                    flat_g = group_idx.view(1, -1).expand(
                        sample_idx_needs.numel(), -1
                    ).reshape(-1)                                    # (Nf*9,)
                    flat_v = adj_fix.reshape(-1, C)                  # (Nf*9, 9)

                    update_tensor = torch.zeros_like(adjusted).index_put(
                        (flat_s, flat_g), flat_v,
                    )
                    update_mask_2d = torch.zeros(
                        B, N, dtype=torch.bool, device=adjusted.device
                    )
                    update_mask_2d[flat_s, flat_g] = True
                    adjusted = torch.where(
                        update_mask_2d.unsqueeze(-1),
                        update_tensor,
                        adjusted,
                    )

                    total_l2 += l2_fix.sum().item()
                    total_edits += int(sample_idx_needs.numel()) * self.grid_size
                    any_edit = True

            if not any_edit:
                break

        info = {
            'total_l2_distance': total_l2,
            'avg_l2_per_sample': total_l2 / max(B, 1),
            'total_constraints_applied': total_edits,
            'n_iterations': n_iter_done,
            'converged': not any_edit,
        }
        return adjusted, info

    def constraint_satisfaction_rate(self, cell_probs):
        """Fraction of (sample, group) pairs where all 9 argmax digits are distinct."""
        B = cell_probs.shape[0]
        preds = cell_probs.argmax(dim=2)                         # (B, 81)
        n_groups = self.groups.shape[0]
        total = 0
        satisfied = 0
        for g in range(n_groups):
            group_preds = preds[:, self.groups[g]]               # (B, 9)
            # unique count == 9 iff all distinct
            # sorted + (diff != 0) counting is simplest in batch:
            sorted_preds, _ = group_preds.sort(dim=1)
            distinct = (sorted_preds[:, 1:] != sorted_preds[:, :-1]).all(dim=1)
            satisfied += distinct.sum().item()
            total += B
        return satisfied / max(total, 1)


if __name__ == "__main__":

    def make_latin_square():
        # Canonical valid Sudoku argmax: value[r,c] = (3*(r%3) + r//3 + c) mod 9.
        # Satisfies AllDifferent on rows, columns, AND 3x3 boxes.
        ls = torch.zeros(81, dtype=torch.long)
        for r in range(9):
            for c in range(9):
                ls[r * 9 + c] = (3 * (r % 3) + r // 3 + c) % 9
        return ls

    def run_suite(pairwise):
        tag = "pairwise" if pairwise else "joint"
        print(f"\n========== SudokuConstraintLayer suite [{tag}] ==========")
        torch.manual_seed(0)
        layer = SudokuConstraintLayer(grid_size=9, pairwise=pairwise)

        # -----------------------------------------------------------------
        # Test 1: targeted conflict injected on uniform-random background.
        # Pairwise MinEdit is a LOCAL greedy conflict-resolver; on uniform-
        # random inputs cascades can chase each other, so we assert only that
        # the forced conflict is resolved, not full sat. The joint path does
        # not suffer the same cycling, but we still only assert the forced
        # conflict here for symmetry.
        # -----------------------------------------------------------------
        B = 2
        probs = torch.softmax(torch.randn(B, 81, 9), dim=2)
        probs[0, 0] = torch.tensor([0.02, 0.02, 0.02, 0.80, 0.02, 0.03, 0.03, 0.03, 0.03])
        probs[0, 1] = torch.tensor([0.02, 0.02, 0.02, 0.60, 0.10, 0.05, 0.05, 0.07, 0.07])

        print(f"[T1 Before] sample 0 cell 0 argmax: {probs[0, 0].argmax().item()}")
        print(f"[T1 Before] sample 0 cell 1 argmax: {probs[0, 1].argmax().item()}")
        adjusted, info = layer(probs, max_iterations=10)
        print(f"[T1 After]  sample 0 cell 0 argmax: {adjusted[0, 0].argmax().item()}")
        print(f"[T1 After]  sample 0 cell 1 argmax: {adjusted[0, 1].argmax().item()}")
        print(f"  edits: {info['total_constraints_applied']}, "
              f"iterations: {info['n_iterations']}")
        assert adjusted[0, 0].argmax().item() != adjusted[0, 1].argmax().item(), \
            "Forced row-0 conflict must be resolved locally"
        print(f"OK (T1/{tag}): forced conflict resolved.")

        # -----------------------------------------------------------------
        # Test 2: trained-network regime — valid Latin square + 1 conflict.
        # -----------------------------------------------------------------
        ls_digits = make_latin_square()                      # (81,) valid
        peaky = torch.full((1, 81, 9), 0.01)
        peaky[0, torch.arange(81), ls_digits] = 0.92
        peaky = peaky / peaky.sum(dim=2, keepdim=True)

        sat_clean = layer.constraint_satisfaction_rate(peaky)
        print(f"[T2 Clean] sat rate on pure Latin square: {sat_clean:.4f}")
        assert sat_clean == 1.0, "Latin square should already satisfy AllDifferent"

        # Inject one conflict in row 0: flip cell 1's argmax to match cell 0.
        d0 = ls_digits[0].item()
        peaky[0, 1] = 0.01
        peaky[0, 1, d0] = 0.80
        peaky[0, 1] = peaky[0, 1] / peaky[0, 1].sum()

        sat_before = layer.constraint_satisfaction_rate(peaky)
        adjusted, info = layer(peaky, max_iterations=5)
        sat_after = layer.constraint_satisfaction_rate(adjusted)
        print(f"[T2] sat: {sat_before:.4f} -> {sat_after:.4f}, "
              f"edits: {info['total_constraints_applied']}, "
              f"iters: {info['n_iterations']}")
        assert sat_after == 1.0, \
            f"Expected 100% sat on near-valid input, got {sat_after}"
        print(f"OK (T2/{tag}): 100% AllDifferent satisfaction on trained-regime input.")

        # -----------------------------------------------------------------
        # Test 3: gradient flow.
        # -----------------------------------------------------------------
        probs = torch.softmax(torch.randn(2, 81, 9), dim=2).requires_grad_(True)
        adjusted, _ = layer(probs)
        loss = -adjusted.log().mean()
        loss.backward()
        assert probs.grad is not None and probs.grad.abs().sum() > 0
        print(f"OK (T3/{tag}): gradient flows through SudokuConstraintLayer.")

    run_suite(pairwise=False)
    run_suite(pairwise=True)
