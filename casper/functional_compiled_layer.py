from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from functional_dag_compile import functional_dag_tables
from minedit import MinEditL2
from signed_clauses import parse_constraints


class FunctionalCompiledConstraintLayer(nn.Module):
    """INFER for constraint sets that are functional DAGs, batched over variables.

    Every non-empty bucket of such a set makes its variable a total function of
    earlier variables (validated by `functional_dag_tables`, which raises
    NotFunctionalDAG otherwise), so S_i(a_<i) is always one value read from a
    lookup table and variables with an empty bucket keep their argmax. The walk
    assigns variables level by level (a variable's level is one past its
    deepest parent); within a level, variables with the same domain and parent
    count are looked up and projected together in one MinEdit call with a
    per-row target, with no host-device synchronisation.

    Produces the same projection, L2 edits and gradients as
    `CompiledConstraintLayer` (the generic per-sample walk) on the same file,
    with the same forward contract: `(B, sum(variable_ranges))` over the
    network's variables; auxiliary variables are fed uniform distributions,
    only their assignments are used, and their edits are excluded.
    """

    def __init__(
        self,
        compiled_path: str | Path,
        variable_ranges: list[int],
        eps: float = 1e-3,
        max_table_entries: int = 100_000,
    ):
        super().__init__()
        cs = parse_constraints(Path(compiled_path).read_text())
        _buckets, tables = functional_dag_tables(cs)
        domains = list(cs.var_domains)

        self.n_real_vars = len(variable_ranges)
        if domains[: self.n_real_vars] != list(variable_ranges):
            raise ValueError(
                f"compiled artifact {compiled_path} declares leading domains "
                f"{domains[: self.n_real_vars]} but the network expects {list(variable_ranges)}"
            )
        self.domains = domains
        self.n_vars = len(domains)
        self.real_total = sum(variable_ranges)
        self.aux_domains = domains[self.n_real_vars:]
        var_starts = [0]
        for h in domains[:-1]:
            var_starts.append(var_starts[-1] + h)

        self._minedits: dict[int, MinEditL2] = {}
        for h in sorted(set(domains[: self.n_real_vars])):
            self._minedits[h] = MinEditL2(h, eps=eps)
            self.add_module(f"_minedit_h{h}", self._minedits[h])

        flat_table: list[int] = []
        table_offset: dict[int, int] = {}
        strides: dict[int, list[int]] = {}
        for var, (parents, mapping) in tables.items():
            size = 1
            for parent in parents:
                size *= domains[parent]
            if size > max_table_entries:
                raise ValueError(
                    f"variable {cs.var_names[var]!r} needs a lookup table with "
                    f"{size} entries (limit {max_table_entries})"
                )
            stride, acc = [], 1
            for parent in reversed(parents):
                stride.append(acc)
                acc *= domains[parent]
            strides[var] = stride[::-1]
            entries = [0] * size
            for key, value in mapping.items():
                entries[sum(k * s for k, s in zip(key, strides[var]))] = value
            table_offset[var] = len(flat_table)
            flat_table.extend(entries)
        self.register_buffer("_table", torch.tensor(flat_table or [0], dtype=torch.long))

        level: dict[int, int] = {}
        for var in cs.ordering:
            parents = tables[var][0] if var in tables else ()
            level[var] = 1 + max((level[p] for p in parents), default=-1)

        groups: dict[tuple[int, int, int, bool], list[int]] = {}
        for var in cs.ordering:
            key = (level[var], domains[var], len(tables[var][0]) if var in tables else -1,
                   var < self.n_real_vars)
            groups.setdefault(key, []).append(var)

        self._groups: list[tuple[int, bool, bool]] = []
        for i, ((_lvl, h, n_parents, real), vars_) in enumerate(sorted(groups.items())):
            prefix = f"_g{i}_"
            self.register_buffer(prefix + "vars", torch.tensor(vars_, dtype=torch.long))
            self.register_buffer(prefix + "cols", torch.tensor(
                [[var_starts[v] + j for j in range(h)] for v in vars_], dtype=torch.long))
            has_table = n_parents >= 0
            if has_table:
                self.register_buffer(prefix + "parents", torch.tensor(
                    [list(tables[v][0]) for v in vars_], dtype=torch.long).reshape(len(vars_), n_parents))
                self.register_buffer(prefix + "strides", torch.tensor(
                    [strides[v] for v in vars_], dtype=torch.long).reshape(len(vars_), n_parents))
                self.register_buffer(prefix + "offset", torch.tensor(
                    [table_offset[v] for v in vars_], dtype=torch.long))
            self._groups.append((i, has_table, real))

    def forward(self, probs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Project `(B, sum(variable_ranges))` probabilities; returns (adjusted, info)."""
        B = probs.shape[0]
        if probs.shape[1] != self.real_total:
            raise ValueError(
                f"expected probs with {self.real_total} columns, got {probs.shape[1]}"
            )
        device = probs.device
        full = torch.cat([probs] + [
            torch.full((B, h), 1.0 / h, device=device, dtype=probs.dtype)
            for h in self.aux_domains
        ], dim=1) if self.aux_domains else probs

        adjusted = full.clone()
        assignments = torch.zeros(B, self.n_vars, dtype=torch.long, device=device)
        l2_per_var = torch.zeros(B, self.n_vars, dtype=probs.dtype, device=device)
        n_projected = torch.zeros((), dtype=torch.long, device=device)

        for i, has_table, real in self._groups:
            vars_ = getattr(self, f"_g{i}_vars")
            cols = getattr(self, f"_g{i}_cols")
            G, h = cols.shape
            p = full[:, cols]
            predicted = p.argmax(dim=2)
            if not has_table:
                assignments[:, vars_] = predicted
                continue

            parents = getattr(self, f"_g{i}_parents")
            flat = (assignments[:, parents] * getattr(self, f"_g{i}_strides")).sum(dim=2)
            target = self._table[flat + getattr(self, f"_g{i}_offset")]
            assignments[:, vars_] = target
            if not real:
                continue

            needs_fix = predicted != target
            proj, l2 = self._minedits[h].project_rows(p.reshape(B * G, h), target.reshape(B * G))
            proj = torch.where(needs_fix.unsqueeze(2), proj.reshape(B, G, h), p)
            l2 = torch.where(needs_fix, l2.reshape(B, G), torch.zeros_like(l2).reshape(B, G))
            adjusted[:, cols.reshape(-1)] = proj.reshape(B, G * h)
            l2_per_var[:, vars_] = l2
            n_projected = n_projected + needs_fix.sum()

        l2_per_sample = l2_per_var[:, : self.n_real_vars].sum(dim=1)
        total_l2, avg_l2, count = torch.stack([
            l2_per_sample.detach().sum(),
            l2_per_sample.detach().mean() if B else l2_per_sample.detach().sum(),
            n_projected.to(probs.dtype),
        ]).tolist()
        info = {
            "projection_loss": l2_per_sample.mean() if B else l2_per_sample.sum(),
            "total_l2_distance": total_l2,
            "avg_l2_per_sample": avg_l2 if B else 0.0,
            "total_constraints_applied": int(count),
            "final_assignments": assignments,
        }
        return adjusted[:, : self.real_total], info
