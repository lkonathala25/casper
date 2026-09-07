import torch
from torch import nn
import torch.nn.functional as F
import itertools


# =============================================================================
# Warcraft path: shared config injected before load_program. ResNetNode and
# WarcraftPathFunctor are registered via SPECIAL_MODELS, which only passes
# ndim into the constructor; extra config is read from this dict.
# =============================================================================
WARCRAFT_CONFIG = {
    "var_ranges": None,        # list[int], length 144
    "total_dim": None,         # int, sum(var_ranges)
    "edge_proj": None,         # (264, total_dim) float tensor
    "edge_endpoints": None,    # (264, 2) long tensor of node ids
    "num_nodes": 144,
    "num_edges": 264,
    "source": 0,
    "sink": 143,
    "num_steps": 22,           # 2*(N-1) for N=12
    "semiring": "max",         # "max" or "sum" (noisy-or)
}


def set_warcraft_config(**kwargs):
    WARCRAFT_CONFIG.update(kwargs)


class SumFunctor(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        result = torch.zeros(9*x.shape[0] + 1).to(x.device)
        inds = torch.meshgrid(
            *tuple([torch.arange(10, device=x.device) for _ in range(x.shape[0])]),
            indexing="ij")
        inds = torch.stack(inds, dim=0).flatten(1)
        xs = torch.stack([x[n_i][i] for n_i, i in enumerate(inds)], dim=0)

        # Compute pairwise products and sum based on i + j
        result.index_add_(0, inds.sum(dim=0), xs.prod(dim=0))


        return result
    
class SumFunctorAll(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        result = torch.zeros(9*x.shape[0] + 1).to(x.device)
        inds = itertools.product(*tuple([list(torch.arange(10))]*x.shape[0]))
        for inds_i in inds:
            t = torch.stack([x[n_i][i] for n_i, i in enumerate(inds_i)], dim=0)
            result[sum(inds_i)] = t.prod(dim=0)
        return result

class AdditionFunctor(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        result = _add_probs(x[0], x[1])
        for t in x[2:]:
            result = _add_probs(result, t)
        return result


def _add_probs(x1, x2):
    result = torch.zeros(10).to(x1.device)
    for i in range(10):
        result += x1[i] * torch.roll(x2, i, 0)
    return result


class CarryFunctor(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        c1 = _carry_probs(x[0], x[1])
        if len(x) == 2:
            return c1
        a1 = _add_probs(x[0], x[1])
        c2 = _carry_probs(a1, x[2])
        result = torch.zeros(10).to(c1.device)
        result[0] = c1[0] * c2[0]
        result[1] = 1 - result[0]
        return result


def _carry_probs(x1, x2):
    result = torch.zeros(10).to(x1.device)
    result[0] = (torch.cumsum(x2, 0).flip((0,)) * x1).sum()
    result[1] = 1 - result[0]
    return result


class EmbeddingFunctor(nn.Module):
    def __init__(self, arity=1, ndims=128):
        super().__init__()
        hidden_dims = max(128, ndims)
        self.model = nn.Sequential(
            nn.Linear(arity * ndims, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.ReLU(True),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(True),
            nn.Linear(hidden_dims, ndims),
        )
        self.activation = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = self.model(x.flatten())
        return self.activation(x)


class LeNet5(nn.Module):
    """
    LeNet5. A small convolutional network.
    """

    def __init__(self, output_features=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # 1 28 28 -> 6 24 24
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, output_features),
        )
        self.activation = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)[0]
        return self.activation(x)


class ResNetNode(nn.Module):
    """ResNet-18 -> per-node categorical (flat, softmax-grouped) for warcraft 12x12.

    Registered as SPECIAL_MODELS[('resnet_node', 1)].

    Input is a stack of one image, shape (1, 3, 96, 96), as produced by
    EmbeddingStore._embed_functor. Output is a flat (total_dim,) tensor
    where each per-node group has been softmaxed independently. The flat
    layout matches build_edge_marginal_projection from the Scallop baseline,
    so WarcraftPathFunctor can multiply by edge_proj directly.
    """

    def __init__(self, *args):
        super().__init__()
        from torchvision import models  # lazy: torchvision import is heavy

        var_ranges = WARCRAFT_CONFIG["var_ranges"]
        total_dim = WARCRAFT_CONFIG["total_dim"]
        assert var_ranges is not None and total_dim is not None, \
            "Call set_warcraft_config(var_ranges=..., total_dim=..., ...) before load_program."

        self.var_ranges = list(var_ranges)
        self.total_dim = int(total_dim)
        self.register_buffer(
            "_group_offsets",
            torch.tensor([0] + list(itertools.accumulate(self.var_ranges))[:-1], dtype=torch.long),
        )
        self.register_buffer("_group_sizes", torch.tensor(self.var_ranges, dtype=torch.long))

        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, self.total_dim)

    def forward(self, x):
        # EmbeddingStore stacks args: x has shape (1, 3, 96, 96).
        if x.dim() == 4 and x.shape[0] == 1:
            img_batch = x
        elif x.dim() == 3:
            img_batch = x.unsqueeze(0)
        else:
            img_batch = x[:1] if x.dim() == 4 else x.view(1, *x.shape[-3:])

        logits = self.backbone(img_batch).squeeze(0)  # (total_dim,)

        # Per-node softmax in-place into a contiguous output of shape (total_dim,).
        out = torch.empty_like(logits)
        offset = 0
        for vr in self.var_ranges:
            out[offset:offset + vr] = F.softmax(logits[offset:offset + vr], dim=0)
            offset += vr
        return out

    @property
    def fc(self):
        # Convenience accessor for diagnostics (grad norm probing).
        return self.backbone.fc


class WarcraftPathFunctor(nn.Module):
    """Differentiable s-t reachability on the 12x12 grid.

    Registered as SPECIAL_MODELS[('warcraft_path', 1)].

    Input: stack of one (total_dim,) per-node categorical, shape (1, total_dim).
    Output: 2-vector [1 - p, p] where p = P(s connected to t under sampled
    edges). With ndim=2 and metric="dot", `eq(~1, ~warcraft_path(...))` then
    yields a proof score of log(p), so NLL loss = -log(p).

    Semiring is read once at construction from WARCRAFT_CONFIG["semiring"]:
      - "max":  reach[v] = max over paths of prod(edge_p along path).
                Mirrors Scallop's difftopkproofs (k=1 / best-proof channel).
      - "sum":  reach[v] = noisy-OR over inflows; double-counts edge-disjoint
                paths but stays in [0, 1]. Smoother gradients.
    """

    def __init__(self, *args):
        super().__init__()
        var_ranges = WARCRAFT_CONFIG["var_ranges"]
        total_dim = WARCRAFT_CONFIG["total_dim"]
        edge_proj = WARCRAFT_CONFIG["edge_proj"]
        edge_endpoints = WARCRAFT_CONFIG["edge_endpoints"]
        for k, v in [("var_ranges", var_ranges), ("total_dim", total_dim),
                     ("edge_proj", edge_proj), ("edge_endpoints", edge_endpoints)]:
            assert v is not None, f"WARCRAFT_CONFIG['{k}'] not set."

        self.num_nodes = int(WARCRAFT_CONFIG["num_nodes"])
        self.num_edges = int(WARCRAFT_CONFIG["num_edges"])
        self.source = int(WARCRAFT_CONFIG["source"])
        self.sink = int(WARCRAFT_CONFIG["sink"])
        self.num_steps = int(WARCRAFT_CONFIG["num_steps"])
        self.semiring = str(WARCRAFT_CONFIG["semiring"])
        assert self.semiring in ("max", "sum"), self.semiring

        self.register_buffer("edge_proj", edge_proj.clone().float())
        self.register_buffer("edge_endpoints", edge_endpoints.clone().long())

    def _edge_probs_from_node_probs(self, node_probs_flat):
        # node_probs_flat: (total_dim,)
        # edge_active[e] = 0.5 * sum_{(n,c)} pattern_c@n[e] * node_probs_flat[off(n)+c]
        return (self.edge_proj @ node_probs_flat) * 0.5  # (264,)

    def _build_adjacency(self, edge_p):
        # Symmetric adjacency from undirected edges.
        A = torch.zeros(self.num_nodes, self.num_nodes, device=edge_p.device, dtype=edge_p.dtype)
        u = self.edge_endpoints[:, 0]
        v = self.edge_endpoints[:, 1]
        A[u, v] = edge_p
        A[v, u] = edge_p
        return A

    def _reach_max(self, A):
        reach = torch.zeros(self.num_nodes, device=A.device, dtype=A.dtype)
        reach[self.source] = 1.0
        for _ in range(self.num_steps):
            # for each v: max_u A[u,v] * reach[u]
            cand = (A * reach.unsqueeze(1)).max(dim=0).values  # (num_nodes,)
            reach = torch.maximum(reach, cand)
        return reach[self.sink]

    def _reach_sum(self, A):
        # Noisy-OR relaxation: at each step, reach[v] <- 1 - prod_u(1 - A[u,v]*reach[u]).
        # Logspace for numerical stability:
        #   log(1 - reach[v]) = sum_u log(1 - A[u,v]*reach[u]).
        eps = 1e-9
        reach = torch.zeros(self.num_nodes, device=A.device, dtype=A.dtype)
        reach[self.source] = 1.0
        for _ in range(self.num_steps):
            inflow = A * reach.unsqueeze(1)              # (u, v)
            log_one_minus = torch.log1p(-inflow.clamp(max=1.0 - eps))
            log_complement = log_one_minus.sum(dim=0)     # (v,)
            new_reach = 1.0 - torch.exp(log_complement)
            reach = torch.maximum(reach, new_reach)
        return reach[self.sink]

    def forward(self, x):
        # EmbeddingStore stacks args: shape (1, total_dim).
        flat = x.squeeze(0) if x.dim() == 2 else x
        edge_p = self._edge_probs_from_node_probs(flat).clamp(0.0, 1.0)
        A = self._build_adjacency(edge_p)
        if self.semiring == "max":
            p = self._reach_max(A)
        else:
            p = self._reach_sum(A)
        # Return [1 - p, p] so dot(one_hot("1"), out) = p.
        out = torch.stack([1.0 - p, p])
        return out


if __name__ == "__main__":
    model = CarryFunctor()
    t = [[0, 0, .2, .2, 0, 0, 0, 0, 0, .6], [0, .8, 0, 0, 0, 0, 0, .1, .1, 0]]
    t = torch.Tensor(t)
    print(model(t))
