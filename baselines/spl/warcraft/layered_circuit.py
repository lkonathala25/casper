from __future__ import annotations

import torch


class LayeredCircuit:
    """SPL's FFCircuit quantities over a pypsdd PSDD, evaluated one layer at a time.

    pypsdd walks the circuit node by node from Python, launching a few tiny
    kernels per node; on the 12x12 steps SDD that is ~360k nodes and ~50 s per
    training batch, almost all of it launch overhead. This engine reads the same
    normalized PSDD (same nodes, same element order), groups decision nodes by
    depth, and evaluates each depth with a handful of batched gathers. The values
    it computes are pypsdd's: Z as generate_tf_ac(log_space=True), the label
    log-likelihood as ff_ll, and the decoded instantiation as parameterize_ff
    followed by get_mpe_inst, including its greedy most-mass branch rule.
    """

    def __init__(self, beta, num_vars: int, device="cuda"):
        nodes = list(beta.positive_iter())
        index = {id(n): i for i, n in enumerate(nodes)}
        n = len(nodes)
        self.pad = n
        self.device = device
        self.num_vars = num_vars
        self.root = index[id(nodes[-1])]

        lit_node, lit_var, lit_pos = [], [], []
        true_node, true_var = [], []
        depth = [0] * n
        decisions = []
        for i, node in enumerate(nodes):
            if node.is_literal():
                lit_node.append(i)
                lit_var.append(abs(node.literal) - 1)
                lit_pos.append(node.literal > 0)
            elif node.is_true():
                true_node.append(i)
                true_var.append(node.vtree.var - 1)
            elif node.is_decomposition():
                elems = [(index[id(p)], index[id(s)]) for p, s in node.positive_elements]
                depth[i] = 1 + max(max(depth[p], depth[s]) for p, s in elems)
                decisions.append((i, elems))
            else:
                raise ValueError(f"unsupported node type in circuit: {node.node_type}")

        def t(x, dtype=torch.long):
            return torch.as_tensor(x, dtype=dtype, device=device)

        self.lit_node, self.lit_var, self.lit_pos = t(lit_node), t(lit_var), t(lit_pos, torch.bool)
        self.true_node, self.true_var = t(true_node), t(true_var)

        by_depth: dict[int, list] = {}
        for i, elems in decisions:
            by_depth.setdefault(depth[i], []).append((i, elems))
        self.layers = []
        for d in sorted(by_depth):
            group = by_depth[d]
            fan = max(len(e) for _, e in group)
            primes = [[p for p, _ in e] + [self.pad] * (fan - len(e)) for _, e in group]
            subs = [[s for _, s in e] + [self.pad] * (fan - len(e)) for _, e in group]
            self.layers.append((t([i for i, _ in group]), t(primes), t(subs)))

    def _leaves(self, neg, pos):
        """(N + 1, B, K) value buffer holding the literal and true-node values."""
        B, _, K = pos.shape
        V = torch.full((self.pad + 1, B, K), float("-inf"), device=self.device, dtype=pos.dtype)
        w = torch.where(self.lit_pos.view(-1, 1, 1),
                        pos[:, self.lit_var].permute(1, 0, 2), neg[:, self.lit_var].permute(1, 0, 2))
        V[self.lit_node] = w
        V[self.true_node] = 0.0
        return V

    def upward(self, neg, pos):
        """Every node's log weighted model count, as pypsdd's generate_tf_ac computes it."""
        V = self._leaves(neg, pos)
        for nodes, primes, subs in self.layers:
            V[nodes] = (V[primes] + V[subs]).logsumexp(dim=1)
        return V

    def log_partition(self, neg, pos):
        """Differentiable log Z of shape (B, K) from (B, V, K) literal log weights."""
        return _LogPartition.apply(neg, pos, self)

    @staticmethod
    def log_likelihood(neg, pos, target):
        """ff_ll of a target that is a model: the sum of its literals' log weights.

        The normalized PSDD is smooth and deterministic, so a model's single
        circuit path carries exactly one literal per variable.
        """
        on = target.unsqueeze(-1) > 0
        return torch.where(on, pos, neg).sum(dim=1)

    def cross_entropy(self, neg, pos, target, mixing):
        """SPL's ff_cross_entropy: -log sum_k pi_k p_k(target), per board."""
        ll = self.log_likelihood(neg, pos, target) - self.log_partition(neg, pos)
        return -(ll + mixing).logsumexp(dim=-1)

    @torch.no_grad()
    def mpe_bits(self, neg, pos, mixing):
        """(B, V) instantiation of SPL's decoder under the argmax mixture component.

        As in pypsdd, each decision node follows its element with the largest
        prime + sub mass (first on ties) and a true node takes its larger literal.
        """
        V = self.upward(neg, pos)
        B = V.shape[1]
        rows = torch.arange(B, device=self.device)
        k = mixing.argmax(dim=-1)
        Vk = V[:, rows, k]
        chosen = torch.zeros((self.pad + 1, B), dtype=torch.bool, device=self.device)
        chosen[self.root] = True
        for nodes, primes, subs in reversed(self.layers):
            best = (Vk[primes] + Vk[subs]).argmax(dim=1)
            active = chosen[nodes]
            node_row, b = active.nonzero(as_tuple=True)
            pick = best[node_row, b]
            chosen[primes[node_row, pick], b] = True
            chosen[subs[node_row, pick], b] = True
        bits = torch.zeros((B, self.num_vars), dtype=torch.long, device=self.device)
        lit_on = chosen[self.lit_node]
        n_idx, b = lit_on.nonzero(as_tuple=True)
        bits[b, self.lit_var[n_idx]] = self.lit_pos[n_idx].long()
        true_on = chosen[self.true_node]
        n_idx, b = true_on.nonzero(as_tuple=True)
        var = self.true_var[n_idx]
        bits[b, var] = (pos[b, var, k[b]] > neg[b, var, k[b]]).long()
        return bits


class _LogPartition(torch.autograd.Function):
    """log Z with a hand-written reverse sweep in place of autograd over every layer."""

    @staticmethod
    def forward(ctx, neg, pos, circuit):
        V = circuit.upward(neg, pos)
        ctx.circuit = circuit
        ctx.save_for_backward(V)
        ctx.shapes = neg.shape
        return V[circuit.root].clone()

    @staticmethod
    def backward(ctx, grad):
        (V,) = ctx.saved_tensors
        c = ctx.circuit
        G = torch.zeros_like(V)
        G[c.root] = grad
        for nodes, primes, subs in reversed(c.layers):
            share = (V[primes] + V[subs] - V[nodes].unsqueeze(1)).exp()
            flow = (G[nodes].unsqueeze(1) * share).flatten(0, 1)
            G.index_add_(0, primes.flatten(), flow)
            G.index_add_(0, subs.flatten(), flow)
        B, num_vars, K = ctx.shapes
        g_lit = G[c.lit_node].permute(1, 0, 2)
        grad_pos = torch.zeros((B, num_vars, K), device=V.device, dtype=V.dtype)
        grad_neg = torch.zeros_like(grad_pos)
        grad_pos.index_add_(1, c.lit_var[c.lit_pos], g_lit[:, c.lit_pos])
        grad_neg.index_add_(1, c.lit_var[~c.lit_pos], g_lit[:, ~c.lit_pos])
        return grad_neg, grad_pos, None
