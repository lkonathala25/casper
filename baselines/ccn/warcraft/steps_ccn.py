import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_ROOT = Path(__file__).resolve().parents[3]
for _d in (_ROOT / "casper", _ROOT / "tasks" / "warcraft", _HERE):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from compile_steps_shield import shield_paths  # noqa: E402
from model import compute_metrics, is_valid_path, node_categorical_loss_vectorised  # noqa: E402
from steps_walk import StepsWalkDecoder, edge_index_maps  # noqa: E402


def req_loss(shield, h, targets, eps=1e-7):
    """ReqLoss: BCE between the goal-conditioned shield output and the one-hot target."""
    reql = shield(h, goal=targets, iterative=False)
    return (-targets * torch.log(reql + eps)
            - (1.0 - targets) * torch.log(1.0 - reql + eps)).mean()


def load_steps_shield(grid_size, at_most_one=False, shield_dir=None, compiled_artifact=None):
    """Load a compiled CCN+ steps shield: (ShieldLayer, atom_of, free_atoms, decoder)."""
    from pishield.propositional_requirements.shield_layer import ShieldLayer
    _, strata_path, atoms_path = shield_paths(grid_size, at_most_one, shield_dir)
    if not strata_path.exists():
        raise FileNotFoundError(
            f"{strata_path} is missing: run baselines/ccn/warcraft/compile_steps_shield.py "
            f"--grid-size {grid_size}{' --at-most-one' if at_most_one else ''} first")
    atom_of = {(i, v): val for i, v, val in json.loads(atoms_path.read_text())}
    free = sorted((val, key) for key, val in atom_of.items() if type(val) is int)
    assert [k for k, _ in free] == list(range(len(free))), "free atoms must be 0..n-1"
    free_atoms = [key for _, key in free]
    with open(strata_path, "rb") as f:
        strata = pickle.load(f)
    shield = ShieldLayer(len(free_atoms), requirements=strata)
    dec = StepsWalkDecoder(grid_size, compiled_artifact=compiled_artifact)
    return shield, atom_of, free_atoms, dec


class StepsLink(torch.nn.Module):
    """Per-cell categorical probabilities -> shield inputs for the free steps atoms.

    The time-aware bridge CASPER's steps layer uses (steps_bridge.TimeAwareBridge),
    written in torch so the ReqLoss gradient reaches the network: the score of cell
    v at tick i is P(v on path) * exp(-|d(s, v) - i|), with P(v on path) =
    1 - P(skip at v), and 1 at the source and sink, which have no skip category.
    """

    def __init__(self, grid_size, node_info, node_order, free_atoms):
        super().__init__()
        skip, off = [], 0
        for node in node_order:
            cats = node_info[node]["categories"]
            skip.append(off + cats.index("skip") if "skip" in cats else -1)
            off += len(cats)
        skip = torch.as_tensor(skip)
        cells = torch.as_tensor([v for _, v in free_atoms])
        ticks = torch.as_tensor([i for i, _ in free_atoms], dtype=torch.float32)
        dist = (cells // grid_size + cells % grid_size).float()
        self.register_buffer("has_skip", skip >= 0)
        self.register_buffer("skip_slots", skip.clamp(min=0))
        self.register_buffer("cell_of_atom", cells)
        self.register_buffer("penalty", torch.exp(-(dist - ticks).abs()))

    def forward(self, probs):
        """(B, total_dim) per-cell probabilities -> (B, n_free) atom scores in [0, 1]."""
        p_skip = probs[:, self.skip_slots]
        onpath = torch.where(self.has_skip, 1.0 - p_skip, torch.ones_like(p_skip))
        return onpath[:, self.cell_of_atom] * self.penalty


def path_sequence(edge_vec, grid_size, horizon):
    """Cells of the s-t path in a binary edge vector, padded with the sink to X_0..X_L."""
    edges, _, _ = edge_index_maps(grid_size)
    adj = {}
    for e in np.flatnonzero(np.asarray(edge_vec) > 0.5):
        r1, c1, r2, c2 = edges[e]
        a, b = r1 * grid_size + c1, r2 * grid_size + c2
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    t = grid_size * grid_size - 1
    seq, prev = [0], None
    while seq[-1] != t:
        nxt = [v for v in adj.get(seq[-1], []) if v != prev]
        if not nxt:
            raise ValueError("edge vector is not a single s-t path")
        prev = seq[-1]
        seq.append(nxt[0])
    return seq + [t] * (horizon + 1 - len(seq))


def goal_atoms(edge_labels, free_index, grid_size, horizon):
    """(B, n_free) targets: the atom (i, X_i) of the ground-truth walk is 1, all others 0."""
    goal = torch.zeros(len(edge_labels), len(free_index))
    for b, vec in enumerate(np.asarray(edge_labels.cpu() if torch.is_tensor(edge_labels) else edge_labels)):
        for i, v in enumerate(path_sequence(vec, grid_size, horizon)):
            k = free_index.get((i, v))
            if k is not None:
                goal[b, k] = 1.0
    return goal


def decode_walks(scores, dec, free_index, atom_of):
    """Greedy walk from s over the feasible sets, taking the highest-scoring atom each tick."""
    out = []
    for row in scores.detach().cpu().numpy():
        seq = [dec.enc.source]
        for i in range(1, dec.enc.horizon + 1):
            cands = dec.feasible(i, seq[-1])
            if not cands:
                seq.append(seq[-1])
                continue
            vals = [row[free_index[(i, c)]] if (i, c) in free_index
                    else float(bool(atom_of.get((i, c)))) for c in cands]
            seq.append(cands[int(np.argmax(vals))])
        out.append(dec.decode(seq)[2])
    return np.stack(out)


def edge_metrics(pred, gt, edges_lookup, grid_size):
    """Exact / Hamming / Consistent for binary edge vectors."""
    pred = np.asarray(pred).astype(np.float32)
    gt = np.asarray(gt).astype(np.float32)
    return {
        "exact": float((pred == gt).all(axis=1).mean() * 100),
        "hamming": float((pred == gt).mean() * 100),
        "consistent": float(np.mean([is_valid_path(p, edges_lookup, grid_size) for p in pred]) * 100),
    }


def train_epoch_steps(model, loader, optimizer, shield, link, var_ranges, free_index,
                      grid_size, horizon, lambda_req, device, grad_clip=0.0):
    """Per-cell CE plus lambda * ReqLoss on the steps atoms."""
    model.train()
    total, n = 0.0, 0
    for imgs, node_labels, edge_labels in tqdm(loader, desc="  Train", leave=False):
        imgs, node_labels = imgs.to(device), node_labels.to(device)
        logits, probs, _, _ = model(imgs)
        goal = goal_atoms(edge_labels, free_index, grid_size, horizon).to(device)
        loss = (node_categorical_loss_vectorised(logits, node_labels, var_ranges)
                + lambda_req * req_loss(shield, link(probs), goal))
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate_steps(model, loader, shield, link, dec, free_index, atom_of, edges,
                   node_info, node_order, var_ranges, grid_size, lambda_req, device):
    """Raw per-cell argmax metrics, and metrics of the walk decoded from the shielded atoms."""
    model.eval()
    total, n, l2 = 0.0, 0, 0.0
    all_probs, all_edges, walks = [], [], []
    for imgs, node_labels, edge_labels in tqdm(loader, desc="  Eval", leave=False):
        imgs, node_labels = imgs.to(device), node_labels.to(device)
        logits, probs, _, _ = model(imgs)
        h = link(probs)
        goal = goal_atoms(edge_labels, free_index, grid_size, dec.enc.horizon).to(device)
        total += (node_categorical_loss_vectorised(logits, node_labels, var_ranges)
                  + lambda_req * req_loss(shield, h, goal)).item()
        n += 1
        # Vectorised forward: identical to pishield's default rule-by-rule one
        # (max |diff| 0 at 12x12, batch 8 and 128) and ~50x faster.
        shielded = shield(h, iterative=False)
        l2 += torch.linalg.vector_norm(shielded - h, ord=2, dim=1).sum().item()
        walks.append(decode_walks(shielded, dec, free_index, atom_of))
        all_probs.append(probs.cpu())
        all_edges.append(edge_labels)
    all_probs, all_edges = torch.cat(all_probs), torch.cat(all_edges)
    raw = compute_metrics(all_probs, all_edges, edges, node_info, node_order,
                          int(edges.shape[0]), grid_size)
    walk = edge_metrics(np.concatenate(walks), all_edges.numpy(), edges, grid_size)
    return {"val_loss": total / max(n, 1),
            "raw_exact": raw["exact"], "raw_hamming": raw["hamming"],
            "raw_consistent": raw["consistent"],
            "exact": walk["exact"], "hamming": walk["hamming"],
            "consistent": walk["consistent"],
            "avg_l2_edit": l2 / all_probs.shape[0], "argmax_changed_frac": float("nan")}
