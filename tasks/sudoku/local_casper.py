from __future__ import annotations

import math

import torch
import torch.nn as nn

from casper_layer import CasperLayer
from constraints import Constraints
from minedit import MinEditL2
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


class JointCasper(nn.Module):
    """One global sweep, conditioning each cell on all three of its groups.

    Visits every cell exactly once in row-major order. The feasible set is the
    intersection of the cell's row, column and box feasible sets, each read
    from the same compiled group template:

        S(cell) = domain \\ (used-in-row  U  used-in-col  U  used-in-box)

    over the cells already visited. Row-major traversal reaches a cell's
    already-visited row, column and box neighbours as a contiguous prefix of
    each group's own ordering, which is what lets one compiled 9-clique serve
    all three lookups.

    Unlike the 27-pass layer this cannot overwrite an earlier correction, so
    there is no last-group-wins effect. The cost is that S can be empty: three
    groups may jointly exhaust the domain. Such a cell is a *dead end* — v1
    keeps its original argmax, records it, and continues. Dead ends are the
    only way a violation can survive, so they are reported, not hidden.
    """

    def __init__(
        self,
        group_constraints: Constraints,
        grid_size: int = 9,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.n_cells = grid_size * grid_size
        self.domain = grid_size
        self.box_side = int(math.isqrt(grid_size))
        if self.box_side * self.box_side != grid_size:
            raise ValueError(f"grid_size {grid_size} is not a perfect square")
        self.constraints = group_constraints
        self.minedit = MinEditL2(grid_size, eps=eps)

    def _coords(self, cell: int) -> tuple[int, int, int, int]:
        """(row, col, box index, position within box) for a cell."""
        g, bs = self.grid_size, self.box_side
        r, c = divmod(cell, g)
        return r, c, (r // bs) * bs + (c // bs), (r % bs) * bs + (c % bs)

    def _box_cells(self, box: int) -> list[int]:
        """The box's cells in its own group ordering (row-major inside the box)."""
        g, bs = self.grid_size, self.box_side
        br, bc = divmod(box, bs)
        return [
            (br * bs + dr) * g + (bc * bs + dc)
            for dr in range(bs)
            for dc in range(bs)
        ]

    def forward(self, probs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Project a batch of boards in one joint sweep.

        Returns the adjusted probs and diagnostics including `dead_ends`, the
        per-sample list of cells whose joint feasible set was empty.
        """
        B, total = probs.shape
        if total != self.n_cells * self.domain:
            raise ValueError(f"expected {self.n_cells * self.domain} columns")

        g = self.grid_size
        board = probs.view(B, self.n_cells, self.domain)
        slices = [board[:, i, :] for i in range(self.n_cells)]

        assigned: list[list[int | None]] = [[None] * self.n_cells for _ in range(B)]
        dead_ends: list[list[int]] = [[] for _ in range(B)]
        l2 = torch.zeros(B, device=probs.device, dtype=probs.dtype)
        n_projected = 0

        for cell in range(self.n_cells):
            r, c, box, bpos = self._coords(cell)
            box_cells = self._box_cells(box)
            cur = slices[cell]
            argmax = cur.argmax(dim=1)

            feasibles: list[frozenset[int]] = []
            for b in range(B):
                row_prior = {j: assigned[b][r * g + j] for j in range(c)}
                col_prior = {i: assigned[b][i * g + c] for i in range(r)}
                box_prior = {p: assigned[b][box_cells[p]] for p in range(bpos)}
                s = (
                    self.constraints.feasible_set(c, row_prior)
                    & self.constraints.feasible_set(r, col_prior)
                    & self.constraints.feasible_set(bpos, box_prior)
                )
                feasibles.append(s)

            groups: dict[frozenset[int], list[int]] = {}
            for b, f in enumerate(feasibles):
                groups.setdefault(f, []).append(b)

            new_slice = cur
            for f, idxs in groups.items():
                if not f:
                    # Dead end: keep the network's own call and record it.
                    for b in idxs:
                        assigned[b][cell] = int(argmax[b].item())
                        dead_ends[b].append(cell)
                    continue

                idx_t = torch.tensor(idxs, device=probs.device, dtype=torch.long)
                preds = argmax[idx_t]
                in_f = torch.zeros(len(idxs), dtype=torch.bool, device=probs.device)
                for t in f:
                    in_f = in_f | (preds == t)
                for j, b in enumerate(idxs):
                    if bool(in_f[j].item()):
                        assigned[b][cell] = int(preds[j].item())

                need = (~in_f).nonzero(as_tuple=False).squeeze(1)
                if need.numel() == 0:
                    continue

                rows = cur[idx_t[need]]
                best_d = best_t = best_a = None
                for t in sorted(f):
                    adj, dist, _ = self.minedit(
                        rows, target_category=int(t), is_positive=True
                    )
                    if best_d is None:
                        best_d, best_a = dist, adj
                        best_t = torch.full(
                            (need.numel(),), int(t),
                            dtype=torch.long, device=probs.device,
                        )
                    else:
                        better = dist < best_d
                        best_d = torch.where(better, dist, best_d)
                        best_t = torch.where(better, torch.full_like(best_t, int(t)), best_t)
                        best_a = torch.where(better.unsqueeze(1), adj, best_a)

                gneed = idx_t[need]
                expanded = torch.zeros(
                    B, self.domain, device=probs.device, dtype=cur.dtype
                ).index_copy(0, gneed, best_a)
                mask = torch.zeros(B, dtype=torch.bool, device=probs.device)
                mask[gneed] = True
                new_slice = torch.where(mask.unsqueeze(1), expanded, new_slice)
                l2 = l2 + torch.zeros(
                    B, device=probs.device, dtype=probs.dtype
                ).index_copy(0, gneed, best_d)
                n_projected += int(need.numel())
                for j, local in enumerate(need.tolist()):
                    assigned[idxs[local]][cell] = int(best_t[j].item())

            slices[cell] = new_slice

        adjusted = torch.stack(slices, dim=1).reshape(B, total)
        info = {
            "projection_loss": l2.sum() / max(B, 1),
            "total_l2_distance": float(l2.sum().item()),
            "total_constraints_applied": n_projected,
            "dead_ends": dead_ends,
            "n_dead_ends": [len(d) for d in dead_ends],
            "final_assignments": [list(a) for a in assigned],
        }
        return adjusted, info


class KPassCasper(nn.Module):
    """K rounds over the 27 blocks, each cell judged against all its peers.

    Every visit computes the cell's feasible set from the *current* values of
    all its row, column and box peers -- not just the ones visited earlier --
    so a cell always sees live state on every side. Each cell is therefore
    reached three times per round (its row block, its column block, its box
    block), and a cell whose feasible set is empty is left alone and retried
    next round rather than being written off.

    The compiled 9-clique serves the all-peers query by relabeling: a clique is
    symmetric under any permutation of its variables, so putting the target
    cell last and its eight peers in positions 0..7 turns "condition on all
    peers" into the complete contiguous prefix the compiled walk requires.

    Stopping rule: a round that makes zero edits *and* hits zero dead ends
    certifies the board. Zero edits alone is not sufficient -- two cells that
    are both dead ends can duplicate each other while neither is editable --
    so `certified` uses the stronger condition and `weak_certified` records
    what the edits-only rule would have claimed.
    """

    def __init__(
        self,
        group_constraints: Constraints,
        grid_size: int = 9,
        max_rounds: int = 10,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.n_cells = grid_size * grid_size
        self.domain = grid_size
        self.max_rounds = max_rounds
        self.box_side = int(math.isqrt(grid_size))
        self.constraints = group_constraints
        self.minedit = MinEditL2(grid_size, eps=eps)
        self.blocks = build_groups(grid_size)
        self._peers = {
            cell: self._peer_lists(cell) for cell in range(self.n_cells)
        }
        # "Everything but v", read from the compiled artifact one peer at a
        # time. Asking it about all eight peers at once instead would answer a
        # different question -- whether the whole group is still extendable --
        # so one duplicate anywhere in a row would report every cell in that
        # row as hopeless. Pairwise queries give the intended "digits no peer
        # currently holds", and go empty only when the peers really do cover
        # the domain. Nine distinct queries exist, so they are cached as masks.
        self._not_mask: list[int] = []
        for v in range(grid_size):
            s = group_constraints.feasible_set(1, {0: v})
            self._not_mask.append(sum(1 << x for x in s))
        self._full_mask = (1 << grid_size) - 1

    def _peer_lists(self, cell: int) -> tuple[list[int], list[int], list[int]]:
        """The cell's eight row peers, eight column peers and eight box peers."""
        g, bs = self.grid_size, self.box_side
        r, c = divmod(cell, g)
        br, bc = (r // bs) * bs, (c // bs) * bs
        row = [r * g + j for j in range(g) if j != c]
        col = [i * g + c for i in range(g) if i != r]
        box = [
            (br + dr) * g + (bc + dc)
            for dr in range(bs)
            for dc in range(bs)
            if (br + dr) * g + (bc + dc) != cell
        ]
        return row, col, box

    def _feasible(self, cell: int, values: list[int]) -> list[int]:
        """Domain minus every value currently held by the cell's peers."""
        m = self._full_mask
        for peers in self._peers[cell]:
            for p in peers:
                m &= self._not_mask[values[p]]
                if not m:
                    return []
        return [v for v in range(self.domain) if m >> v & 1]

    def forward(self, probs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Run up to `max_rounds` rounds, stopping when every board certifies."""
        B, total = probs.shape
        if total != self.n_cells * self.domain:
            raise ValueError(f"expected {self.n_cells * self.domain} columns")

        board = probs.view(B, self.n_cells, self.domain)
        slices = [board[:, i, :].clone() for i in range(self.n_cells)]
        assigned = [
            [int(v) for v in board[b].argmax(dim=1).tolist()] for b in range(B)
        ]

        certified: list[int | None] = [None] * B
        weak_certified: list[int | None] = [None] * B
        edits_curve: list[list[int]] = []
        dead_curve: list[list[int]] = []
        l2 = torch.zeros(B, device=probs.device, dtype=probs.dtype)
        rounds_used = 0

        for rnd in range(self.max_rounds):
            rounds_used = rnd + 1
            r_edits = [0] * B
            r_dead = [0] * B

            for _name, cells in self.blocks:
                for cell in cells:
                    for b in range(B):
                        if certified[b] is not None:
                            continue
                        s = self._feasible(cell, assigned[b])
                        if not s:
                            r_dead[b] += 1
                            continue
                        if assigned[b][cell] in s:
                            continue
                        row = slices[cell][b : b + 1]
                        best_d = best_t = best_a = None
                        for t in s:
                            adj, dist, _ = self.minedit(
                                row, target_category=int(t), is_positive=True
                            )
                            if best_d is None or float(dist[0]) < float(best_d[0]):
                                best_d, best_a, best_t = dist, adj, int(t)
                        slices[cell] = slices[cell].clone()
                        slices[cell][b] = best_a[0]
                        l2[b] = l2[b] + best_d[0]
                        assigned[b][cell] = best_t
                        r_edits[b] += 1

            edits_curve.append(list(r_edits))
            dead_curve.append(list(r_dead))
            for b in range(B):
                if certified[b] is None:
                    if r_edits[b] == 0 and weak_certified[b] is None:
                        weak_certified[b] = rnd + 1
                    if r_edits[b] == 0 and r_dead[b] == 0:
                        certified[b] = rnd + 1
            if all(c is not None for c in certified):
                break

        adjusted = torch.stack(slices, dim=1).reshape(B, total)
        info = {
            "projection_loss": l2.sum() / max(B, 1),
            "total_l2_distance": float(l2.sum().item()),
            "rounds_used": rounds_used,
            "certified_at": certified,
            "weak_certified_at": weak_certified,
            "edits_per_round": edits_curve,
            "dead_ends_per_round": dead_curve,
            "final_assignments": [list(a) for a in assigned],
        }
        return adjusted, info


def alldiff_group_constraints(size: int = 9) -> Constraints:
    """Closed-form ALLDIFFERENT feasibility for one group, no compile needed.

    S_i(a_<i) is just the domain minus the values the group's earlier cells
    already took. Used as an independent oracle against the compiled artifact.
    """

    def feasible(step: int, prior_assignment: dict[int, int], var_domain: int):
        # Mirror the compiled artifact exactly. Its clause walk is defined only
        # for a complete contiguous prefix, and it reports an unsatisfiable
        # group -- a prefix that already repeats a value -- as an empty set
        # rather than as "anything unused". Both cases matter once joint
        # conditioning lets an inconsistent prefix reach a later cell.
        if set(prior_assignment) != set(range(step)):
            return []
        used = list(prior_assignment.values())
        if len(set(used)) != len(used):
            return []
        taken = set(used)
        return [v for v in range(var_domain) if v not in taken]

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


class GroupKPassCasper(KPassCasper):
    """KPassCasper with group-local feasibility (8 peers of the current group).

    Group-local feasibility: while block `cells` is processed, a cell's
    feasible set is the domain minus the digits currently held by the other 8
    cells of *that* group, read pairwise from the compiled 9-clique through
    the inherited `_not_mask` lookups. KPassCasper instead intersects all 24
    row, column and box peers at once.

    Tolerated cross-group violations: a cell may therefore take a digit that
    clashes with its other two groups. The clash is repaired when those groups
    are processed later in the round (or in a later round). This is what lets
    multi-cell repair chains go through that the 24-peer rule blocks. Because
    8 peers cannot cover 9 digits, the feasible set is never empty, so there
    are no dead ends.

    Visiting order: with cell_order="confidence" (default) the cells of a
    group are visited least-confident first, by the current probability of
    their assigned digit, so the less trusted cell of a duplicate pair is the
    one that moves. cell_order="canonical" keeps the group's own order.

    Unchanged from KPassCasper: the single-cell MinEditL2 positive-head
    projection onto the nearest feasible digit, `max_rounds`, and the stopping
    rule (a board is certified after a round with zero edits).

    Ported verbatim from llm-sudoku/sudoku/src/casper_sudoku_layer.py; the
    bit-identical parity test is tasks/sudoku/test_group_kpass.py.
    """

    def __init__(self, *args, cell_order='confidence', **kwargs):
        super().__init__(*args, **kwargs)
        if cell_order not in ('canonical', 'confidence'):
            raise ValueError(f'cell_order must be canonical or confidence, got {cell_order!r}')
        self.cell_order = cell_order

    def _feasible_in_group(self, cell, cells, values):
        m = self._full_mask
        for p in cells:
            if p != cell:
                m &= self._not_mask[values[p]]
        return [v for v in range(self.domain) if m >> v & 1]

    def forward(self, probs):
        B, total = probs.shape
        if total != self.n_cells * self.domain:
            raise ValueError(f'expected {self.n_cells * self.domain} columns, got {total}')

        board = probs.view(B, self.n_cells, self.domain)
        slices = [board[:, i, :].clone() for i in range(self.n_cells)]
        assigned = [[int(v) for v in board[b].argmax(dim=1).tolist()] for b in range(B)]

        # A round with zero edits means every group is a permutation, so the
        # board is valid; group-local feasible sets are never empty (8 peers
        # cannot cover 9 digits), so there are no dead ends to track.
        certified = [None] * B
        edits_curve = []
        l2 = torch.zeros(B, device=probs.device, dtype=probs.dtype)
        rounds_used = 0

        for rnd in range(self.max_rounds):
            rounds_used = rnd + 1
            r_edits = [0] * B
            for _name, cells in self.blocks:
                for b in range(B):
                    if certified[b] is not None:
                        continue
                    order = list(cells)
                    if self.cell_order == 'confidence':
                        conf = {c: slices[c][b, assigned[b][c]].item() for c in cells}
                        order.sort(key=lambda c: conf[c])
                    for cell in order:
                        s = self._feasible_in_group(cell, cells, assigned[b])
                        if assigned[b][cell] in s:
                            continue
                        row = slices[cell][b:b + 1]
                        best_d = best_t = best_a = None
                        for t in s:
                            adj, dist, _ = self.minedit(row, target_category=int(t), is_positive=True)
                            if best_d is None or dist[0].item() < best_d[0].item():
                                best_d, best_a, best_t = dist, adj, int(t)
                        slices[cell] = slices[cell].clone()
                        slices[cell][b] = best_a[0]
                        l2[b] = l2[b] + best_d[0]
                        assigned[b][cell] = best_t
                        r_edits[b] += 1
            edits_curve.append(list(r_edits))
            for b in range(B):
                if certified[b] is None and r_edits[b] == 0:
                    certified[b] = rnd + 1
            if all(c is not None for c in certified):
                break

        adjusted = torch.stack(slices, dim=1).reshape(B, total)
        return adjusted, {
            'projection_loss': l2.sum() / max(B, 1),
            'total_l2_distance': float(l2.sum().item()),
            'rounds_used': rounds_used,
            'certified_at': certified,
            'edits_per_round': edits_curve,
            'final_assignments': [list(a) for a in assigned],
        }
