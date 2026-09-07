import logging
import sys
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
# Appended, not prepended: src/ and src_warcraft/ both define constraint_layer,
# data, model, minedit and train, so putting src/ first silently shadows this
# package's own modules.
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from constraints import Constraints
from signed_clauses import SignedClause, SignedLiteral

from edge_bridge import CategoricalEdgeBridge, default_artifact
from masked_loss import vectorised_positive_head

logger = logging.getLogger(__name__)


def _load_sibling(name):
    """Import a module from this directory by path, ignoring sys.path order."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"_warcraft_{name}", Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RingCapExceeded(Exception):
    """A board still had stray rings after the round cap."""


class WarcraftLazyExactLayer(nn.Module):
    """Degree projection plus lazy elimination of stray rings.

    Family 1 (per-node degree clauses) is compiled and loaded; it admits a
    source-sink path together with any number of vertex-disjoint rings. Each
    round walks from the source, and if the walk fails to cover every on-edge
    the uncovered edges form rings. Every such ring contributes a clause saying
    at least one of its edges is off, and the board is re-projected.

    The only exit is the walk-and-count certificate: walked on-edges equal to
    total on-edges. Boards still uncertified at the round cap are reported as
    failures, never as passes.
    """

    def __init__(self, grid_size=12, artifact=None, alphabet='full', eps=1e-3,
                 max_rounds=20):
        super().__init__()
        self.grid_size = grid_size
        self.eps = eps
        self.max_rounds = max_rounds
        self.artifact_path = Path(artifact) if artifact else default_artifact(grid_size)
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"constraint artifact not found: {self.artifact_path}")

        self.cons = Constraints.from_compiled_dsl(self.artifact_path)
        self.bridge = CategoricalEdgeBridge(grid_size, alphabet=alphabet)
        self.num_edges = self.bridge.num_edges
        self.total_dim = self.bridge.total_dim
        self.var_ranges = self.bridge.var_ranges

        if len(self.cons.ordering) != self.num_edges:
            raise ValueError(
                f"artifact has {len(self.cons.ordering)} variables but the "
                f"{grid_size}x{grid_size} grid has {self.num_edges} edges")

        self._ordering = list(self.cons.ordering)
        self._step_of = {var: step for step, var in enumerate(self._ordering)}

        self.source = (0, 0)
        self.sink = (grid_size - 1, grid_size - 1)
        self._endpoints = {}
        for node in self.bridge.node_order:
            for e in self.bridge.node_info[node]['all_edges']:
                self._endpoints.setdefault(e, []).append(node)
        self._endpoints = {e: tuple(v) for e, v in self._endpoints.items()}

        self._groups = defaultdict(list)
        off = 0
        for n_idx, vr in enumerate(self.var_ranges):
            self._groups[vr].append((n_idx, off))
            off += vr

        self.lifetime_empty_feasible = 0
        self.lifetime_boards = 0
        self.lifetime_boards_with_empty = 0
        self.lifetime_capped_boards = 0
        self.lifetime_rings_forbidden = 0

    # ---- feasibility -----------------------------------------------------

    def _apply_extra(self, feasible, clauses, var, assignment):
        """Intersect `feasible` with per-board ring clauses at this step."""
        for clause in clauses:
            satisfied = False
            for lit in clause.literals:
                if lit.var == var:
                    continue
                if lit.var in assignment and assignment[lit.var] in lit.values:
                    satisfied = True
                    break
            if satisfied:
                continue
            feasible = feasible & (clause.get(var) or frozenset())
            if not feasible:
                break
        return feasible

    def _project_one(self, row, overlay):
        """Greedy walk over the elimination ordering for one board."""
        assignment = {}
        out = [0] * self.num_edges
        empty = forced = 0
        sites = []
        for step, var in enumerate(self._ordering):
            fs = self.cons.feasible_set(step, assignment)
            if overlay:
                extra = overlay.get(step)
                if extra:
                    fs = self._apply_extra(fs, extra, var, assignment)
            p_on = float(row[var])
            if not fs:
                # The overlay made this step infeasible. Fall back to the base
                # clauses so the board stays degree-legal; the round simply
                # fails to certify and the loop tries again.
                empty += 1
                if len(sites) < 20:
                    sites.append((step, var))
                base_fs = self.cons.feasible_set(step, assignment)
                fs = base_fs if base_fs else frozenset({1 if p_on >= 0.5 else 0})
            if len(fs) == 1:
                forced += 1
                val = next(iter(fs))
            else:
                val = 1 if (1 in fs and p_on >= 0.5) else (
                    0 if 0 in fs else next(iter(fs)))
            assignment[var] = val
            out[var] = val
        return out, empty, forced, sites

    # ---- certificate -----------------------------------------------------

    def _adjacency(self, on_edges):
        adj = defaultdict(list)
        for e in on_edges:
            a, b = self._endpoints[e]
            adj[a].append((b, e))
            adj[b].append((a, e))
        return adj

    def walk_certificate(self, on_edges):
        """Walk from the source; returns (walked_edges, certified)."""
        on_edges = set(on_edges)
        adj = self._adjacency(on_edges)
        walked = set()
        cur = self.source
        while True:
            nxt = [(n, e) for n, e in adj.get(cur, ()) if e not in walked]
            if not nxt:
                break
            node, e = nxt[0]
            walked.add(e)
            cur = node
        return walked, (len(walked) == len(on_edges))

    def find_rings(self, on_edges, walked):
        """Trace each stray ring among the on-edges the source walk missed."""
        remaining = set(on_edges) - set(walked)
        adj = self._adjacency(remaining)
        rings = []
        while remaining:
            seed = min(remaining)
            start = self._endpoints[seed][0]
            ring, cur = [], start
            while True:
                nxt = [(n, e) for n, e in adj.get(cur, ()) if e in remaining]
                if not nxt:
                    break
                node, e = nxt[0]
                ring.append(e)
                remaining.discard(e)
                cur = node
                if cur == start:
                    break
            if not ring:
                remaining.discard(seed)
                continue
            rings.append(ring)
        return rings

    def validate(self, row):
        """Independent validity check, deliberately not reusing the walk.

        Degrees are recounted from scratch and connectivity is decided by BFS
        over the on-edge subgraph rather than by the sequential trace.
        """
        violations = []
        on = [e for e, v in enumerate(row) if v]
        if any(e < 0 or e >= self.num_edges for e in on):
            violations.append("edge index outside the grid")
        if not on:
            violations.append("no on-edges")
            return violations

        deg = defaultdict(int)
        for e in on:
            a, b = self._endpoints[e]
            deg[a] += 1
            deg[b] += 1

        if deg[self.source] != 1:
            violations.append(f"source degree {deg[self.source]} != 1")
        if deg[self.sink] != 1:
            violations.append(f"sink degree {deg[self.sink]} != 1")
        for node, d in deg.items():
            if node in (self.source, self.sink):
                continue
            if d not in (0, 2):
                violations.append(f"node {node} degree {d} not in {{0,2}}")

        adj = self._adjacency(on)
        touched = {n for n, d in deg.items() if d > 0}
        seen = {self.source}
        queue = deque([self.source])
        while queue:
            cur = queue.popleft()
            for n, _ in adj.get(cur, ()):
                if n not in seen:
                    seen.add(n)
                    queue.append(n)
        if seen != touched:
            violations.append(
                f"on-edges span {len(touched)} nodes but only {len(seen)} "
                f"reachable from the source ({len(touched - seen)} stranded)")
        return violations

    # ---- the loop --------------------------------------------------------

    def project_board(self, row):
        """Run the refinement loop for a single board's edge scores."""
        overlay = defaultdict(list)
        log = {"rounds": 0, "rings_per_round": [], "edges_flipped_per_round": [],
               "empty_feasible": 0, "forced_steps": 0, "empty_sites": [],
               "capped": False, "violations": []}
        prev = None
        assign = None
        # Working copy: ring edges are damped so the greedy walk moves off a
        # forbidden ring instead of rediscovering it. The appended clause is what
        # makes the loop correct and terminating; this only steers the search.
        row = [float(x) for x in row]

        for rnd in range(1, self.max_rounds + 1):
            assign, empty, forced, sites = self._project_one(row, overlay)
            log["rounds"] = rnd
            log["empty_feasible"] += empty
            log["forced_steps"] += forced
            log["empty_sites"].extend(sites[:5])
            flipped = 0 if prev is None else sum(
                1 for a, b in zip(prev, assign) if a != b)
            log["edges_flipped_per_round"].append(flipped)
            prev = list(assign)

            on = [e for e, v in enumerate(assign) if v]
            walked, certified = self.walk_certificate(on)
            if certified:
                log["rings_per_round"].append(0)
                break

            rings = self.find_rings(on, walked)
            log["rings_per_round"].append(len(rings))
            self.lifetime_rings_forbidden += len(rings)
            for ring in rings:
                clause = SignedClause.from_literals(
                    [SignedLiteral(e, frozenset({0})) for e in ring])
                step = max(self._step_of[e] for e in ring)
                overlay[step].append(clause)
                for e in ring:
                    row[e] *= 0.1
        else:
            log["capped"] = True
            self.lifetime_capped_boards += 1

        log["violations"] = self.validate(assign)
        log["certified"] = (not log["capped"]) and not log["violations"]
        return assign, log

    def push_to_categories(self, probs, target_cats):
        """MinEdit L2 push so each node's argmax becomes its target category."""
        adjusted = probs.clone()
        targets = torch.as_tensor(target_cats, dtype=torch.long, device=probs.device)
        n_edits = 0
        B = probs.shape[0]

        for vr, node_list in self._groups.items():
            slices = [adjusted[:, off:off + vr] for _, off in node_list]
            tgts = [targets[:, n_idx] for n_idx, _ in node_list]
            p_adj, _, n_active = vectorised_positive_head(
                torch.cat(slices, dim=0), torch.cat(tgts, dim=0), eps=self.eps)
            n_edits += int((n_active > 0).sum())
            for k, (_, off) in enumerate(node_list):
                adjusted[:, off:off + vr] = p_adj[k * B:(k + 1) * B]

        return adjusted, n_edits

    def forward(self, probs, max_iterations=1):
        scores = self.bridge.edge_scores(probs).detach().cpu()
        B = probs.shape[0]

        edge_assign = torch.zeros(B, self.num_edges, dtype=torch.int64)
        logs = []
        empty_total = boards_empty = forced = 0
        capped = []
        violating = []
        sites = []

        for b in range(B):
            assign, log = self.project_board(scores[b])
            edge_assign[b] = torch.tensor(assign, dtype=torch.int64)
            logs.append(log)
            empty_total += log["empty_feasible"]
            boards_empty += 1 if log["empty_feasible"] else 0
            forced += log["forced_steps"]
            sites.extend(log["empty_sites"][:5])
            if log["capped"]:
                capped.append(b)
            elif log["violations"]:
                # A capped board is already a reported failure; a board that
                # certified yet fails the independent check is an invariant breach.
                violating.append((b, log["violations"]))

        self.lifetime_empty_feasible += empty_total
        self.lifetime_boards += B
        self.lifetime_boards_with_empty += boards_empty

        if empty_total:
            logger.warning(
                "WarcraftLazyExactLayer: %d empty feasible set(s) across %d "
                "board(s) of %d", empty_total, boards_empty, B)
        if capped:
            logger.error(
                "WarcraftLazyExactLayer: %d board(s) hit the %d-round cap "
                "without certifying: %s", len(capped), self.max_rounds, capped[:10])

        # Halt on any violation: a board must never leave the loop invalid.
        if violating:
            b, why = violating[0]
            raise RingCapExceeded(
                f"{len(violating)} board(s) exited the loop failing the "
                f"independent validity check; first is board {b}: {why}")

        target_cats = self.bridge.edges_to_categories(edge_assign)
        adjusted, n_edits = self.push_to_categories(probs, target_cats)

        rounds = [lg["rounds"] for lg in logs]
        info = {
            "empty_feasible_sets": empty_total,
            "boards_with_empty_feasible": boards_empty,
            "empty_sites": sites,
            "forced_steps": forced,
            "total_steps": sum(lg["rounds"] for lg in logs) * self.num_edges,
            "lifetime_empty_feasible": self.lifetime_empty_feasible,
            "lifetime_boards": self.lifetime_boards,
            "lifetime_boards_with_empty": self.lifetime_boards_with_empty,
            "lifetime_capped_boards": self.lifetime_capped_boards,
            "lifetime_rings_forbidden": self.lifetime_rings_forbidden,
            "cells_edited": n_edits,
            "avg_corrections_per_sample": n_edits / max(B, 1),
            "edge_assignment": edge_assign,
            "per_board": logs,
            "rounds_max": max(rounds) if rounds else 0,
            "rounds_mean": sum(rounds) / max(len(rounds), 1),
            "boards_needing_refinement": sum(1 for r in rounds if r > 1),
            "capped_boards": capped,
            "certified_boards": sum(1 for lg in logs if lg["certified"]),
        }
        return adjusted, info


# Backwards-compatible name: workstreams 3/4 wired against this class before the
# ring loop existed, and the degree projection is still its first round.
WarcraftEdgeConstraintLayer = WarcraftLazyExactLayer


def build_constraint_layer(mode, grid_size=12, processing_order='row',
                           artifact=None, alphabet='full', max_rounds=20):
    """Select the projection layer.

    'monotone'   — the original directional flow sweep (down/right only).
    'lazy_exact' — compiled per-edge degree clauses plus ring elimination.
    """
    if mode == 'monotone':
        legacy = _load_sibling("constraint_layer")
        return legacy.WarcraftConstraintLayer(grid_size=grid_size,
                                              processing_order=processing_order)
    if mode == 'lazy_exact':
        return WarcraftLazyExactLayer(grid_size=grid_size, artifact=artifact,
                                      alphabet=alphabet, max_rounds=max_rounds)
    raise ValueError(f"unknown constraint mode {mode!r}")
