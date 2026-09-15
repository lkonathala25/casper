"""
Scallop-constrained Warcraft 12x12 baseline.

Pipeline:
  ResNet-18 -> per-node categorical (CASPER's head, tasks/warcraft/model.py)
            -> marginalise to per-edge active probabilities
            -> Scallop program over the edge facts
            -> P(valid_path)

--constraint steps (default): the steps walk at(i, v) over the domains of
CASPER's compiled steps artifact, each move along a probabilistic edge, the sink
absorbing, valid_path = at(L, sink). --constraint reach: the original transitive
closure in warcraft_path.scl, valid_path = path(0, 143).

Loss = per-node CE  +  lambda_path * (-log P(valid_path)). The program is used
only in this training loss; inference is per-node argmax.
"""

import argparse
import csv
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import scallopy
import sys

# Shared Warcraft task modules (categories, dataset, backbone, loss, metrics)
# live in tasks/warcraft of this repo, alongside CASPER.
_TASK_DIR = Path(__file__).resolve().parents[3] / "tasks" / "warcraft"
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))

from node_categories import build_node_categories  # noqa: E402
from data import WarcraftDataset  # noqa: E402
from model import (  # noqa: E402
    ResNet18NodeCategorical,
    node_categorical_loss_vectorised,
    compute_metrics as _compute_metrics,
)


def compute_metrics(node_probs, edge_labels, edges, node_info, node_order):
    """Shared Exact / Hamming / Consistent on this grid."""
    return _compute_metrics(node_probs, edge_labels, edges, node_info,
                            node_order, NUM_EDGES, GRID)


EPS = 1e-8
GRID = 12
NUM_NODES = GRID * GRID            # 144
NUM_EDGES = 264
SOURCE_ID = 0                      # (0, 0)
SINK_ID = NUM_NODES - 1            # (11, 11)


# =============================================================================
# Per-node categorical -> per-edge active probability
# =============================================================================

def build_edge_marginal_projection(node_info, node_order, num_edges=NUM_EDGES):
    """
    (num_edges, total_dim) sum-projection P such that
        edge_active[b, e] = sum_{(n, c) : pattern_c at node n activates e}
                                node_probs[b, offset(n) + c].
    Each edge has exactly two endpoint nodes contributing, so we divide by 2
    in node_probs_to_edge_probs (averaging the two endpoint marginals).
    """
    total_dim = sum(info["n_cats"] for info in node_info.values())
    proj = torch.zeros(num_edges, total_dim)
    offset = 0
    for node in node_order:
        info = node_info[node]
        for cat_idx, pattern in enumerate(info["edge_patterns"]):
            for edge_idx, value in pattern.items():
                if value == 1:
                    proj[edge_idx, offset + cat_idx] = 1.0
        offset += info["n_cats"]
    return proj


def node_probs_to_edge_probs(node_probs, proj):
    return (node_probs @ proj.t()) * 0.5


def steps_program(enc):
    """Scallop program for the steps encoding, built from a compiled steps artifact.

    at(i, v) holds when the walk can be at cell v at tick i. Each move follows a
    probabilistic edge fact into D_{i+1}; the sink absorbs without an edge; the
    query is X_L = sink. dom and nxt are certain facts, so the only
    probabilistic inputs are the edges, exactly as in warcraft_path.scl.
    """
    s, t, L = enc.source, enc.target, enc.horizon
    dom = ", ".join(f"({i}, {v})" for i, d in enumerate(enc.domains) for v in d)
    nxt = ", ".join(f"({i}, {i + 1})" for i in range(L))
    return f"""
rel dom = {{{dom}}}
rel nxt = {{{nxt}}}
rel undirected(u, v) = edge(u, v)
rel undirected(u, v) = edge(v, u)
rel at(0, {s})
rel at(j, v) = at(i, u), nxt(i, j), undirected(u, v), dom(j, v), u != {t}
rel at(j, {t}) = at(i, {t}), nxt(i, j), dom(j, {t})
rel valid_path(0) = at({L}, {t})
"""


# =============================================================================
# Model
# =============================================================================

class ScallopWarcraftNet(nn.Module):
    def __init__(self, var_ranges, node_info, node_order, edges,
                 provenance, k, dispatch, scl_path=None, program_text=None):
        super().__init__()
        self.backbone = ResNet18NodeCategorical(var_ranges, grid_size=GRID,
                                                constraint_mode='none')

        proj = build_edge_marginal_projection(node_info, node_order)
        self.register_buffer("proj", proj)

        edge_pairs = [
            (int(r1) * GRID + int(c1), int(r2) * GRID + int(c2))
            for r1, c1, r2, c2 in edges
        ]
        assert len(edge_pairs) == NUM_EDGES, \
            f"edges.npy has {len(edge_pairs)} rows, expected {NUM_EDGES}"

        self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
        self.scl_ctx.add_relation("edge", (int, int), input_mapping=edge_pairs)
        if program_text is not None:
            self.scl_ctx.add_program(program_text)
        else:
            self.scl_ctx.import_file(str(scl_path))
        self.path_fn = self.scl_ctx.forward_function(
            "valid_path", output_mapping=[(0,)], dispatch=dispatch)

    def forward(self, x):
        logits, node_probs, _, _ = self.backbone(x)
        edge_probs = node_probs_to_edge_probs(node_probs, self.proj)
        # Scallop forward returns shape (batch, 1).
        path_prob = self.path_fn(edge=edge_probs).squeeze(-1)
        return logits, node_probs, edge_probs, path_prob


# =============================================================================
# Train / eval
# =============================================================================

def train_one_epoch(model, loader, optimizer, var_ranges,
                    lambda_path, grad_clip, device):
    model.train()
    total_ce = total_path = 0.0
    n = 0
    for imgs, node_labels, _ in tqdm(loader, desc="  Train", leave=False):
        imgs = imgs.to(device)
        node_labels = node_labels.to(device)

        optimizer.zero_grad()
        logits, _, _, path_prob = model(imgs)
        ce_loss = node_categorical_loss_vectorised(logits, node_labels, var_ranges)

        # Scallop output may live on CPU; align with ce_loss device.
        path_prob = path_prob.to(ce_loss.device)
        path_loss = -torch.log(path_prob + EPS).mean()

        loss = ce_loss + lambda_path * path_loss
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_ce += ce_loss.item()
        total_path += path_loss.item()
        n += 1
    return total_ce / max(n, 1), total_path / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, edges, node_info, node_order, var_ranges, device):
    model.eval()
    total_ce = 0.0
    n = 0
    all_probs, all_edge_labels = [], []
    for imgs, node_labels, edge_labels in tqdm(loader, desc="  Eval", leave=False):
        imgs = imgs.to(device)
        node_labels = node_labels.to(device)
        # Skip the Scallop forward at eval — metrics only need node_probs.
        logits, node_probs, _, _ = model.backbone(imgs)
        ce_loss = node_categorical_loss_vectorised(logits, node_labels, var_ranges)
        total_ce += ce_loss.item()
        n += 1
        all_probs.append(node_probs.cpu())
        all_edge_labels.append(edge_labels)
    all_probs = torch.cat(all_probs, dim=0)
    all_edge_labels = torch.cat(all_edge_labels, dim=0)
    metrics = compute_metrics(all_probs, all_edge_labels, edges, node_info, node_order)
    metrics["loss"] = total_ce / max(n, 1)
    return metrics


# =============================================================================
# Run loop
# =============================================================================

def run(args, device, train_loader, val_loader, test_loader,
        edges, node_info, node_order, var_ranges, scl_path, program_text=None):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"scallop_{args.n_runs}runs.csv"

    # Self-describing run record: the k-ablation report reads this rather than
    # re-parsing hyperparameters out of directory names.
    with open(save_dir / "run_config.json", "w") as cf:
        json.dump(vars(args), cf, indent=2)

    # train_time_s is the k-sensitive quantity: eval skips the Scallop forward
    # entirely, so epoch_time_s alone understates how cost scales with k.
    header = ["run_id", "epoch", "k", "train_ce", "train_path", "val_loss",
              "exact", "hamming", "consistent",
              "epoch_time_s", "train_time_s", "eval_time_s",
              "gpu_mem_peak_mb", "cpu_rss_peak_mb",
              "lr"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        f.flush()

        for run_id in range(args.n_runs):
            seed = args.seed + run_id
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model = ScallopWarcraftNet(
                var_ranges, node_info, node_order, edges,
                provenance=args.provenance,
                k=args.k,
                dispatch=args.dispatch,
                scl_path=scl_path,
                program_text=program_text,
            ).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)

            if args.scheduler == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
            elif args.scheduler == "plateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=5)
            else:
                scheduler = None

            print(f"\n--- Run {run_id + 1}/{args.n_runs} (seed={seed}) ---")
            best_val_exact = -float("inf")
            patience_counter = 0

            for epoch in range(1, args.epochs + 1):
                t0 = time.time()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)

                tr_ce, tr_path = train_one_epoch(
                    model, train_loader, optimizer, var_ranges,
                    args.lambda_path, args.grad_clip, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t_train = time.time() - t0

                val = evaluate(model, val_loader, edges, node_info, node_order,
                               var_ranges, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t_eval = time.time() - t0 - t_train

                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val["loss"])
                elif scheduler is not None:
                    scheduler.step()
                lr = optimizer.param_groups[0]["lr"]
                dt = time.time() - t0
                if device.type == "cuda":
                    gpu_mem_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                else:
                    gpu_mem_peak_mb = float("nan")
                # ru_maxrss is kilobytes on Linux. Lifetime peak (monotonically
                # non-decreasing across epochs).
                cpu_rss_peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

                writer.writerow([
                    run_id, epoch, args.k, tr_ce, tr_path, val["loss"],
                    val["exact"], val["hamming"], val["consistent"],
                    dt, t_train, t_eval, gpu_mem_peak_mb, cpu_rss_peak_mb,
                    lr,
                ])
                f.flush()

                print(f"  Ep {epoch:3d} | CE: {tr_ce:.4f} | "
                      f"Path: {tr_path:.4f} | "
                      f"VLoss: {val['loss']:.4f} | "
                      f"Ex/Ham/Con: "
                      f"{val['exact']:5.1f}/{val['hamming']:5.1f}/{val['consistent']:5.1f}"
                      f" | {dt:.1f}s (tr {t_train:.1f}s) | "
                      f"gpu_peak={gpu_mem_peak_mb:.0f}MB cpu_peak={cpu_rss_peak_mb:.0f}MB"
                      f" | LR: {lr:.2e}")

                if val["exact"] > best_val_exact:
                    best_val_exact = val["exact"]
                    patience_counter = 0
                    torch.save(model.state_dict(), save_dir / f"best_run{run_id}.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f"  Early stopping at epoch {epoch} "
                              f"(best val_exact={best_val_exact:.2f})")
                        break

            model.load_state_dict(torch.load(
                save_dir / f"best_run{run_id}.pt", weights_only=True))
            test = evaluate(model, test_loader, edges, node_info, node_order,
                            var_ranges, device)
            print(f"  Test | Ex/Ham/Con: "
                  f"{test['exact']:5.1f}/{test['hamming']:5.1f}/{test['consistent']:5.1f}")
            with open(save_dir / f"test_run{run_id}.json", "w") as tf:
                json.dump(test, tf, indent=2)

    print(f"\nResults -> {csv_path}")


# =============================================================================
# Entry
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to preprocessed_4conn directory.")
    parser.add_argument("--save_dir", type=str, default="results/warcraft_scallop")
    parser.add_argument("--scl_path", type=str, default=None,
                        help="Path to .scl program. Defaults to "
                             "warcraft_path.scl alongside this script.")
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau", "none"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-path", type=float, default=0.1,
                        help="Weight on the Scallop path-constraint loss term.")
    parser.add_argument("--provenance", type=str, default="difftopkproofs")
    parser.add_argument("--k", type=int, default=3,
                        help="Top-k proofs retained per derived fact.")
    parser.add_argument("--dispatch", type=str, default="parallel",
                        choices=["single", "serial", "parallel"])
    parser.add_argument("--alphabet", type=str, default="full",
                        choices=["monotone", "full"],
                        help="Per-cell category alphabet. 'monotone' cannot "
                             "represent paths that move up or left.")
    parser.add_argument("--train-limit", type=int, default=None,
                        help="Train on the first N boards only (smoke tests).")
    parser.add_argument("--constraint", type=str, default="steps",
                        choices=["steps", "reach"],
                        help="'steps': walk X_0..X_L over the compiled steps domains "
                             "(same artifact as CASPER); 'reach': the original "
                             "transitive-closure program in --scl_path.")
    parser.add_argument("--steps-artifact", type=str, default=None,
                        help="Compiled steps artifact. Defaults to constraints/"
                             "warcraft/steps_artifacts/steps_<GRID>_trimmed_compiled.json.")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    node_info, var_ranges, node_order, total_dim = build_node_categories(
        GRID, alphabet=args.alphabet)
    print(f"Nodes: {len(node_order)}, total_dim: {total_dim}, alphabet: {args.alphabet}")

    data_dir = Path(args.data_dir)
    train_ds = WarcraftDataset(data_dir, "train", node_info, node_order)
    val_ds = WarcraftDataset(data_dir, "val", node_info, node_order)
    test_ds = WarcraftDataset(data_dir, "test", node_info, node_order)
    if args.train_limit:
        train_ds = Subset(train_ds, range(args.train_limit))
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    edges = np.load(data_dir / "edges.npy")
    assert edges.shape[0] == NUM_EDGES, \
        f"Expected {NUM_EDGES} edges in edges.npy, got {edges.shape[0]}"

    program_text = None
    scl_path = Path(args.scl_path) if args.scl_path else Path(__file__).parent / "warcraft_path.scl"
    if args.constraint == "steps":
        from steps_walk import load_compiled_encoding
        art = Path(args.steps_artifact) if args.steps_artifact else (
            _TASK_DIR.parents[1] / "constraints" / "warcraft" / "steps_artifacts"
            / f"steps_{GRID}_trimmed_compiled.json")
        enc = load_compiled_encoding(art)
        if enc.g != GRID:
            raise ValueError(f"{art} is for a {enc.g}x{enc.g} grid, not {GRID}x{GRID}")
        program_text = steps_program(enc)
        print(f"Scallop constraint: steps walk X_0..X_{enc.horizon} over {art.name} "
              f"({sum(len(d) for d in enc.domains)} (tick, cell) pairs)")
    else:
        if not scl_path.exists():
            raise FileNotFoundError(f"Scallop program not found: {scl_path}")
        print(f"Scallop program: {scl_path}")
    print(f"Provenance: {args.provenance} (k={args.k}), dispatch: {args.dispatch}")
    print(f"Lambda path: {args.lambda_path}")

    run(args, device, train_loader, val_loader, test_loader,
        edges, node_info, node_order, var_ranges, scl_path,
        program_text=program_text)


if __name__ == "__main__":
    main()
