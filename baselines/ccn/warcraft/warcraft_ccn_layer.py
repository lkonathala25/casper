"""CCN+ (pishield) constraint layer for Warcraft shortest-path routing.

Variables:
    For each node in row-major order, that node's routing categories are laid
    out contiguously in the variable vector. Variable y_{s+k} is 1 iff node
    n has category k active, where s = start_indices[n]. Total dim matches
    build_node_categories(grid_size).

Propositional constraints (emitted as a pishield .txt):
    1. At-least-one per node:           y_{s+0} or ... or y_{s+K-1}
    2. At-most-one per node:            not y_{s+a} or not y_{s+b}   (a<b)
    3. Flow consistency per edge e:
         For each edge, let
           E_u = {cats at the emitter endpoint u with pattern[e]=1}
           E_v = {cats at the acceptor endpoint v with pattern[e]=1}
         Emit bi-implication (OR E_u) <-> (OR E_v) as:
           for each a in E_u:  not y_{u+a} or (OR_{b in E_v} y_{v+b})
           for each b in E_v:  not y_{v+b} or (OR_{a in E_u} y_{u+a})

Source/sink have no 'skip' category, so (1) alone already pins them to emit
(resp. absorb) exactly one edge.
"""

from pathlib import Path

import sys

from pishield.shield_layer import build_shield_layer

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TASK_DIR = _REPO_ROOT / "tasks" / "warcraft"
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))

from node_categories import build_node_categories  # noqa: E402


def default_constraint_file(grid_size, alphabet):
    """Where the generated CCN+ requirements file for this grid lives."""
    return (_REPO_ROOT / "constraints" / "warcraft" / "ccn"
            / f"warcraft_ccn_{grid_size}x{grid_size}_{alphabet}.txt")


def _compute_start_indices(var_ranges):
    starts = []
    off = 0
    for vr in var_ranges:
        starts.append(off)
        off += vr
    return starts


def _edge_endpoints(grid_size):
    """Return {edge_idx: ((i_u,j_u), (i_v,j_v))}.

    u is the emitter (pattern has outgoing e set); v is the acceptor.
    Indexing matches tasks/warcraft/node_categories.build_node_categories.
    """
    n_w = grid_size - 1
    n_right = grid_size * n_w
    endpoints = {}
    for i in range(grid_size):
        for j in range(grid_size):
            if j < grid_size - 1:
                endpoints[i * n_w + j] = ((i, j), (i, j + 1))
            if i < grid_size - 1:
                endpoints[n_right + i * grid_size + j] = ((i, j), (i + 1, j))
    return endpoints


def _clause_str(literals):
    return " or ".join(literals)


def generate_warcraft_constraints(grid_size=12, filepath=None, alphabet="full"):
    """Emit a propositional constraint file for Warcraft shortest-path routing.

    Returns (filepath, num_variables, clause_count).
    """
    if filepath is None:
        filepath = default_constraint_file(grid_size, alphabet)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    filepath = str(filepath)

    node_info, var_ranges, node_order, total_dim = build_node_categories(
        grid_size, alphabet=alphabet)
    start_indices = _compute_start_indices(var_ranges)
    node_idx = {node: i for i, node in enumerate(node_order)}

    clause_count = 0
    with open(filepath, "w") as f:
        for n_idx, node in enumerate(node_order):
            n_cats = node_info[node]["n_cats"]
            s = start_indices[n_idx]
            f.write(_clause_str([f"y_{s + k}" for k in range(n_cats)]) + "\n")
            clause_count += 1

        for n_idx, node in enumerate(node_order):
            n_cats = node_info[node]["n_cats"]
            s = start_indices[n_idx]
            for a in range(n_cats):
                for b in range(a + 1, n_cats):
                    f.write(f"not y_{s + a} or not y_{s + b}\n")
                    clause_count += 1

        endpoints = _edge_endpoints(grid_size)
        for e, (u_node, v_node) in endpoints.items():
            u_i = node_idx[u_node]
            v_i = node_idx[v_node]
            s_u = start_indices[u_i]
            s_v = start_indices[v_i]

            E_u = [k for k, p in enumerate(node_info[u_node]["edge_patterns"])
                   if p.get(e, 0) == 1]
            E_v = [k for k, p in enumerate(node_info[v_node]["edge_patterns"])
                   if p.get(e, 0) == 1]

            if not E_u or not E_v:
                continue

            v_lits = [f"y_{s_v + k}" for k in E_v]
            for a in E_u:
                f.write(_clause_str([f"not y_{s_u + a}"] + v_lits) + "\n")
                clause_count += 1

            u_lits = [f"y_{s_u + k}" for k in E_u]
            for b in E_v:
                f.write(_clause_str([f"not y_{s_v + b}"] + u_lits) + "\n")
                clause_count += 1

    return filepath, total_dim, clause_count


def build_warcraft_shield(grid_size=12, constraint_file=None,
                          ordering_choice="rev-degree", alphabet="full"):
    """Build pishield constraint layer for Warcraft routing.

    If `constraint_file` is None, generates it at the default location.
    """
    if constraint_file is None:
        constraint_file = default_constraint_file(grid_size, alphabet)

    if not Path(constraint_file).exists():
        generate_warcraft_constraints(grid_size=grid_size, filepath=constraint_file,
                                      alphabet=alphabet)

    _, _, _, total_dim = build_node_categories(grid_size, alphabet=alphabet)

    return build_shield_layer(
        requirements_filepath=str(constraint_file),
        num_variables=total_dim,
        ordering_choice=ordering_choice,
        requirements_type="propositional",
    )


def node_labels_to_onehot(node_labels, var_ranges, device=None):
    """(batch, num_nodes) int64 category indices -> (batch, total_dim) one-hot.

    Matches the variable layout used by the shield.
    """
    import torch
    if device is None:
        device = node_labels.device
    batch = node_labels.shape[0]
    total_dim = sum(var_ranges)
    onehot = torch.zeros(batch, total_dim, device=device)
    offset = 0
    for n_idx, vr in enumerate(var_ranges):
        onehot[torch.arange(batch, device=device), offset + node_labels[:, n_idx]] = 1.0
        offset += vr
    return onehot


if __name__ == "__main__":
    import torch

    print("Generating 12x12 Warcraft CCN+ constraints...")
    fp, n_vars, n_clauses = generate_warcraft_constraints(grid_size=12)
    print(f"  file: {fp}")
    print(f"  variables: {n_vars}")
    print(f"  clauses: {n_clauses}")

    print("\nBuilding pishield shield...")
    shield = build_warcraft_shield(grid_size=12, constraint_file=fp)
    print(f"  shield built: {type(shield).__name__}")

    _, var_ranges, _, total_dim = build_node_categories(12, alphabet="full")
    probs = torch.zeros(2, total_dim)
    offset = 0
    for vr in var_ranges:
        probs[:, offset:offset + vr] = 1.0 / vr
        offset += vr
    out = shield(probs)
    print(f"  shield(uniform) shape: {tuple(out.shape)}, "
          f"min={out.min().item():.4f}, max={out.max().item():.4f}")

    offset = 0
    multi_active = 0
    for vr in var_ranges:
        block = out[0, offset:offset + vr]
        if (block > 0.5).sum().item() > 1:
            multi_active += 1
        offset += vr
    print(f"  sample-0 nodes with >1 cat >0.5 after shield: {multi_active}")
