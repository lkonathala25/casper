import torch
import torch.nn as nn
from minedit import MinEditL2
from node_categories import build_node_categories


class WarcraftConstraintLayer(nn.Module):

    def __init__(self, grid_size=12, processing_order='row'):
        super().__init__()
        self.grid_size = grid_size
        if processing_order not in ('row', 'diagonal'):
            raise ValueError(f"processing_order must be 'row' or 'diagonal', got {processing_order!r}")
        self.processing_order = processing_order

        self.node_info, self.var_ranges, self.node_order, self.total_dim = \
            build_node_categories(grid_size, alphabet='monotone')

        self.start_indices = []
        offset = 0
        for vr in self.var_ranges:
            self.start_indices.append(offset)
            offset += vr

        self.minedit_modules = nn.ModuleDict({
            str(n): MinEditL2(n_categories=n)
            for n in set(self.var_ranges)
        })

        self._precompute_flow_maps()
        self._precompute_tensors()

        # Iteration order over node_order positions. node_order itself stays
        # row-major so _left_idx / _above_idx (which assume i*N+j layout) keep
        # working — only the visit order changes.
        self._processing_order = self._build_processing_order(processing_order)

        print(f"WarcraftConstraintLayer (node-categorical, batch-vectorised) initialized:")
        print(f"  Grid: {grid_size}x{grid_size}")
        print(f"  Nodes: {len(self.node_order)}")
        print(f"  Output dim: {self.total_dim}")
        print(f"  Processing order: {processing_order}")
        print(f"  MinEdit modules: {list(self.minedit_modules.keys())}")

    def _build_processing_order(self, mode):
        """List of node-positions (indices into self.node_order).

        - 'row': identity (top-left → bottom-right, row by row).
        - 'diagonal': anti-diagonal sweep (i+j ascending), top-down tiebreak (i ascending).
          Both are valid topological orders for the R/D-edge DAG.
        """
        if mode == 'row':
            return list(range(len(self.node_order)))
        positions = list(range(len(self.node_order)))
        positions.sort(key=lambda p: (self.node_order[p][0] + self.node_order[p][1],
                                       self.node_order[p][0]))
        return positions

    def _precompute_flow_maps(self):
        self.flow_maps = {}

        for node in self.node_order:
            info = self.node_info[node]
            cats = info['categories']
            incoming = info['incoming']
            outgoing = info['outgoing']

            accepting_R, accepting_D, emitting_R, emitting_D = [], [], [], []
            skip_cat = None

            for cat_idx, (name, pattern) in enumerate(zip(cats, info['edge_patterns'])):
                if name == 'skip':
                    skip_cat = cat_idx
                if 'R' in incoming and pattern.get(incoming['R'], 0) == 1:
                    accepting_R.append(cat_idx)
                if 'D' in incoming and pattern.get(incoming['D'], 0) == 1:
                    accepting_D.append(cat_idx)
                if 'R' in outgoing and pattern.get(outgoing['R'], 0) == 1:
                    emitting_R.append(cat_idx)
                if 'D' in outgoing and pattern.get(outgoing['D'], 0) == 1:
                    emitting_D.append(cat_idx)

            self.flow_maps[node] = {
                'accepting_R': accepting_R,
                'accepting_D': accepting_D,
                'emitting_R': emitting_R,
                'emitting_D': emitting_D,
                'skip_cat': skip_cat,
            }

    def _precompute_tensors(self):
        N = self.grid_size

        self._emitting_R_sets = []
        self._emitting_D_sets = []
        self._accepting_R_sets = []
        self._accepting_D_sets = []
        self._skip_cats = []
        self._is_source = []
        self._is_sink = []

        for node_idx, node in enumerate(self.node_order):
            fm = self.flow_maps[node]
            ni = self.node_info[node]
            self._emitting_R_sets.append(set(fm['emitting_R']))
            self._emitting_D_sets.append(set(fm['emitting_D']))
            self._accepting_R_sets.append(set(fm['accepting_R']))
            self._accepting_D_sets.append(set(fm['accepting_D']))
            self._skip_cats.append(fm['skip_cat'])
            self._is_source.append(ni['is_source'])
            self._is_sink.append(ni['is_sink'])

        self._left_idx = []
        self._above_idx = []
        for node_idx, (i, j) in enumerate(self.node_order):
            self._left_idx.append(i * N + (j - 1) if j > 0 else -1)
            self._above_idx.append((i - 1) * N + j if i > 0 else -1)

    def _get_batch_node_preds(self, probs, node_idx):
        start = self.start_indices[node_idx]
        n_cats = self.var_ranges[node_idx]
        return probs[:, start:start + n_cats].argmax(dim=1)

    def _get_batch_node_probs(self, probs, node_idx):
        start = self.start_indices[node_idx]
        n_cats = self.var_ranges[node_idx]
        return probs[:, start:start + n_cats]

    def _batch_positive_head(self, probs, node_idx, target_cats, mask):
        if not mask.any():
            return 0

        start = self.start_indices[node_idx]
        n_cats = self.var_ranges[node_idx]
        minedit = self.minedit_modules[str(n_cats)]
        eps = minedit.eps

        indices = mask.nonzero(as_tuple=True)[0]
        count = 0

        for idx in indices:
            i = idx.item()
            p = probs[i, start:start + n_cats]
            target = target_cats[i].item()

            p_adj = p.clone()
            active = []
            for j in range(n_cats):
                if j != target and p[target] < p[j] + eps:
                    active.append(j)

            if active:
                n_active = len(active)
                active_sum = sum(p[j] for j in active)
                pool = p[target] + active_sum
                p_adj[target] = (pool + n_active * eps) / (1 + n_active)
                for j in active:
                    p_adj[j] = (pool - eps) / (1 + n_active)
                probs[i, start:start + n_cats] = p_adj
                count += 1

        return count

    def _batch_negative_head(self, probs, node_idx, forbidden_cat, mask):
        if not mask.any():
            return 0

        start = self.start_indices[node_idx]
        n_cats = self.var_ranges[node_idx]

        indices = mask.nonzero(as_tuple=True)[0]
        node_probs = probs[indices, start:start + n_cats].clone()
        node_probs[:, forbidden_cat] = -float('inf')
        runner_ups = node_probs.argmax(dim=1)

        target_cats_full = torch.zeros(probs.shape[0], dtype=torch.long, device=probs.device)
        target_cats_full[indices] = runner_ups
        return self._batch_positive_head(probs, node_idx, target_cats_full, mask)

    def forward(self, probs, max_iterations=1):
        batch_size = probs.shape[0]
        adjusted = probs.clone()
        device = probs.device

        total_corrections = 0

        for iteration in range(max_iterations):
            iter_corrections = 0

            for node_idx in self._processing_order:
                node = self.node_order[node_idx]
                i, j = node

                if self._is_source[node_idx]:
                    continue

                current_preds = self._get_batch_node_preds(adjusted, node_idx)

                flow_from_R = torch.zeros(batch_size, dtype=torch.bool, device=device)
                left_idx = self._left_idx[node_idx]
                if left_idx >= 0:
                    left_preds = self._get_batch_node_preds(adjusted, left_idx)
                    for cat in self._emitting_R_sets[left_idx]:
                        flow_from_R |= (left_preds == cat)

                flow_from_D = torch.zeros(batch_size, dtype=torch.bool, device=device)
                above_idx = self._above_idx[node_idx]
                if above_idx >= 0:
                    above_preds = self._get_batch_node_preds(adjusted, above_idx)
                    for cat in self._emitting_D_sets[above_idx]:
                        flow_from_D |= (above_preds == cat)

                both_flow = flow_from_R & flow_from_D
                if both_flow.any() and left_idx >= 0 and above_idx >= 0:
                    left_probs = self._get_batch_node_probs(adjusted, left_idx)
                    above_probs = self._get_batch_node_probs(adjusted, above_idx)
                    left_preds_b = self._get_batch_node_preds(adjusted, left_idx)
                    above_preds_b = self._get_batch_node_preds(adjusted, above_idx)

                    left_conf = left_probs.gather(1, left_preds_b.unsqueeze(1)).squeeze(1)
                    above_conf = above_probs.gather(1, above_preds_b.unsqueeze(1)).squeeze(1)

                    keep_R = both_flow & (left_conf >= above_conf)
                    if keep_R.any():
                        above_node_probs = self._get_batch_node_probs(adjusted, above_idx)
                        non_emitting_D = [c for c in range(self.var_ranges[above_idx])
                                          if c not in self._emitting_D_sets[above_idx]]
                        if non_emitting_D:
                            non_D_probs = above_node_probs[:, non_emitting_D]
                            best_alt_local = non_D_probs.argmax(dim=1)
                            best_alt = torch.tensor([non_emitting_D[x] for x in best_alt_local.tolist()],
                                                    device=device)
                            iter_corrections += self._batch_positive_head(
                                adjusted, above_idx, best_alt, keep_R)

                    keep_D = both_flow & (above_conf > left_conf)
                    if keep_D.any():
                        non_emitting_R = [c for c in range(self.var_ranges[left_idx])
                                          if c not in self._emitting_R_sets[left_idx]]
                        if non_emitting_R:
                            left_node_probs = self._get_batch_node_probs(adjusted, left_idx)
                            non_R_probs = left_node_probs[:, non_emitting_R]
                            best_alt_local = non_R_probs.argmax(dim=1)
                            best_alt = torch.tensor([non_emitting_R[x] for x in best_alt_local.tolist()],
                                                    device=device)
                            iter_corrections += self._batch_positive_head(
                                adjusted, left_idx, best_alt, keep_D)

                    flow_from_D = flow_from_D & ~keep_R
                    flow_from_R = flow_from_R & ~keep_D

                current_preds = self._get_batch_node_preds(adjusted, node_idx)

                if self._is_sink[node_idx]:
                    for flow_flag, accepting_set in [
                        (flow_from_R & ~flow_from_D, self._accepting_R_sets[node_idx]),
                        (flow_from_D & ~flow_from_R, self._accepting_D_sets[node_idx]),
                    ]:
                        if not flow_flag.any() or not accepting_set:
                            continue
                        valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                        for cat in accepting_set:
                            valid_mask |= (current_preds == cat)
                        needs_fix = flow_flag & ~valid_mask
                        if needs_fix.any():
                            node_probs = self._get_batch_node_probs(adjusted, node_idx)
                            accept_probs = node_probs[:, list(accepting_set)]
                            best_local = accept_probs.argmax(dim=1)
                            accept_list = list(accepting_set)
                            best_cat = torch.tensor([accept_list[x] for x in best_local.tolist()],
                                                    device=device)
                            iter_corrections += self._batch_positive_head(
                                adjusted, node_idx, best_cat, needs_fix)
                else:
                    has_flow = flow_from_R | flow_from_D

                    for flow_flag, accepting_set in [
                        (flow_from_R & ~flow_from_D, self._accepting_R_sets[node_idx]),
                        (flow_from_D & ~flow_from_R, self._accepting_D_sets[node_idx]),
                    ]:
                        if not flow_flag.any() or not accepting_set:
                            continue
                        valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                        for cat in accepting_set:
                            valid_mask |= (current_preds == cat)
                        needs_fix = flow_flag & ~valid_mask
                        if needs_fix.any():
                            node_probs = self._get_batch_node_probs(adjusted, node_idx)
                            accept_probs = node_probs[:, list(accepting_set)]
                            best_local = accept_probs.argmax(dim=1)
                            accept_list = list(accepting_set)
                            best_cat = torch.tensor([accept_list[x] for x in best_local.tolist()],
                                                    device=device)
                            iter_corrections += self._batch_positive_head(
                                adjusted, node_idx, best_cat, needs_fix)

                    skip_cat = self._skip_cats[node_idx]
                    if skip_cat is not None:
                        no_flow = ~has_flow
                        not_skip = current_preds != skip_cat
                        needs_skip = no_flow & not_skip
                        if needs_skip.any():
                            skip_target = torch.full((batch_size,), skip_cat,
                                                     dtype=torch.long, device=device)
                            iter_corrections += self._batch_positive_head(
                                adjusted, node_idx, skip_target, needs_skip)

            total_corrections += iter_corrections
            if iter_corrections == 0:
                break

        return adjusted, {
            'total_corrections': total_corrections,
            'avg_corrections_per_sample': total_corrections / max(batch_size, 1),
        }
