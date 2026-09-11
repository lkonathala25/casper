import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[3]
for _d in (_ROOT / "tasks" / "warcraft", Path(__file__).resolve().parent):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from data import WarcraftDataset  # noqa: E402
from model import ResNet18NodeCategorical, compute_metrics  # noqa: E402
from node_categories import build_node_categories  # noqa: E402
from warcraft_ccn_layer import build_warcraft_shield, node_labels_to_onehot  # noqa: E402
from steps_ccn import (  # noqa: E402
    StepsLink, evaluate_steps, load_steps_shield, req_loss, train_epoch_steps,
)


def ccn_loss(shield, h, targets, eps):
    """ReqLoss on the per-cell category variables (the local encoding)."""
    return req_loss(shield, h, targets, eps)


def train_epoch_ccn(model, loader, optimizer, shield, var_ranges, device,
                    grad_clip=0.0, eps=1e-7):
    """CCN+ training: sigmoid(logits) -> shield(goal=onehot) -> BCE."""
    model.train()
    total_loss, n = 0.0, 0
    for imgs, node_labels, _ in tqdm(loader, desc="  Train", leave=False):
        imgs, node_labels = imgs.to(device), node_labels.to(device)
        logits = model.backbone(imgs)
        targets = node_labels_to_onehot(node_labels, var_ranges, device=device)
        loss = ccn_loss(shield, torch.sigmoid(logits), targets, eps)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_ccn(model, loader, edges, node_info, node_order, var_ranges,
                 shield, device, grid_size, eps=1e-7):
    """CCN+ eval: raw sigmoid and shielded probabilities, each decoded per-cell by argmax."""
    model.eval()
    total_loss, n, total_l2, argmax_changed, cells = 0.0, 0, 0.0, 0, 0
    all_raw, all_proj, all_edge_labels = [], [], []
    for imgs, node_labels, edge_labels in tqdm(loader, desc="  Eval", leave=False):
        imgs, node_labels = imgs.to(device), node_labels.to(device)
        h = torch.sigmoid(model.backbone(imgs))
        targets = node_labels_to_onehot(node_labels, var_ranges, device=device)
        total_loss += ccn_loss(shield, h, targets, eps).item()
        n += 1
        shielded = shield(h)
        all_raw.append(h.cpu())
        all_proj.append(shielded.cpu())
        all_edge_labels.append(edge_labels)
        total_l2 += torch.linalg.vector_norm(shielded - h, ord=2, dim=1).sum().item()
        off = 0
        for vr in var_ranges:
            argmax_changed += (h[:, off:off + vr].argmax(1)
                               != shielded[:, off:off + vr].argmax(1)).sum().item()
            off += vr
        cells += h.shape[0] * len(var_ranges)

    all_raw, all_proj = torch.cat(all_raw), torch.cat(all_proj)
    all_edge_labels = torch.cat(all_edge_labels)
    num_edges = int(edges.shape[0])
    raw = compute_metrics(all_raw, all_edge_labels, edges, node_info, node_order,
                          num_edges, grid_size)
    proj = compute_metrics(all_proj, all_edge_labels, edges, node_info, node_order,
                           num_edges, grid_size)
    return {
        "val_loss": total_loss / max(n, 1),
        "raw_exact": raw["exact"], "raw_hamming": raw["hamming"],
        "raw_consistent": raw["consistent"],
        "exact": proj["exact"], "hamming": proj["hamming"],
        "consistent": proj["consistent"],
        "avg_l2_edit": total_l2 / all_raw.shape[0],
        "argmax_changed_frac": argmax_changed / max(cells, 1),
    }


def main():
    """Train and test CCN+ (pishield) on Warcraft with the per-cell categorical head.

    --constraint steps (default) shields the steps atoms X_i = v of a precompiled
    steps shield (compile_steps_shield.py), fed by the time-aware bridge; --constraint
    local shields the per-cell categories with exactly-one and edge-agreement clauses.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="results/warcraft_ccn")
    p.add_argument("--grid-size", type=int, default=12)
    p.add_argument("--alphabet", type=str, default="full", choices=["monotone", "full"])
    p.add_argument("--constraint", type=str, default="steps", choices=["steps", "local"])
    p.add_argument("--ccn-constraint-file", type=str, default=None,
                   help="Local requirements file; generated under constraints/warcraft/ccn/ if absent.")
    p.add_argument("--at-most-one", action="store_true",
                   help="Steps: use the shield compiled with pairwise at-most-one per tick.")
    p.add_argument("--shield-dir", type=str, default=None,
                   help="Steps: directory holding the compiled strata (default constraints/warcraft/ccn).")
    p.add_argument("--steps-artifact", type=str, default=None,
                   help="Steps: compiled steps artifact for the decoder (default for this grid).")
    p.add_argument("--lambda-req", type=float, default=1.0,
                   help="Steps: weight of ReqLoss on the steps atoms, added to per-cell CE.")
    p.add_argument("--n-runs", type=int, default=2)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Default 64 for steps (the goal-conditioned shield needs about "
                        "0.14 GB per board at 12x12, so 128 exceeds a 24 GB GPU), 128 for local.")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--train-limit", type=int, default=None,
                   help="Train on the first N boards only (smoke tests).")
    args = p.parse_args()
    if args.batch_size is None:
        args.batch_size = 64 if args.constraint == "steps" else 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = args.grid_size
    node_info, var_ranges, node_order, total_dim = build_node_categories(
        N, alphabet=args.alphabet)
    print(f"Device: {device} | grid {N}x{N} | alphabet={args.alphabet} | total_dim={total_dim}")

    data_dir = Path(args.data_dir)
    train_ds = WarcraftDataset(data_dir, "train", node_info, node_order)
    val_ds = WarcraftDataset(data_dir, "val", node_info, node_order)
    test_ds = WarcraftDataset(data_dir, "test", node_info, node_order)
    if args.train_limit:
        train_ds = Subset(train_ds, range(args.train_limit))
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    edges = np.load(data_dir / "edges.npy")

    t0 = time.time()
    if args.constraint == "steps":
        art = Path(args.steps_artifact) if args.steps_artifact else (
            _ROOT / "constraints" / "warcraft" / "steps_artifacts"
            / f"steps_{N}_trimmed_compiled.json")
        shield, atom_of, free_atoms, dec = load_steps_shield(
            N, args.at_most_one, args.shield_dir, art if art.exists() else None)
        shield = shield.to(device)
        link = StepsLink(N, node_info, node_order, free_atoms).to(device)
        free_index = {key: k for k, key in enumerate(free_atoms)}
        print(f"Steps shield loaded in {time.time() - t0:.1f}s: {len(free_atoms)} free atoms, "
              f"{len(shield.strata)} strata")

        def train_fn(model, opt):
            """One epoch of CE + lambda * ReqLoss on the steps atoms."""
            return train_epoch_steps(model, train_loader, opt, shield, link, var_ranges,
                                     free_index, N, dec.enc.horizon, args.lambda_req,
                                     device, grad_clip=args.grad_clip)

        def eval_fn(model, loader):
            """Raw per-cell metrics and metrics of the decoded shielded walk."""
            return evaluate_steps(model, loader, shield, link, dec, free_index, atom_of,
                                  edges, node_info, node_order, var_ranges, N,
                                  args.lambda_req, device)
    else:
        shield = build_warcraft_shield(grid_size=N, constraint_file=args.ccn_constraint_file,
                                       alphabet=args.alphabet).to(device)
        print(f"Local shield built in {time.time() - t0:.1f}s")

        def train_fn(model, opt):
            """One epoch of ReqLoss on the per-cell category variables."""
            return train_epoch_ccn(model, train_loader, opt, shield, var_ranges, device,
                                   grad_clip=args.grad_clip)

        def eval_fn(model, loader):
            """Raw and shielded per-cell metrics."""
            return evaluate_ccn(model, loader, edges, node_info, node_order, var_ranges,
                                shield, device, N)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    header = ["run_id", "epoch", "train_loss", "val_loss", "raw_exact", "raw_hamming",
              "raw_consistent", "exact", "hamming", "consistent", "avg_l2_edit",
              "argmax_changed_frac", "epoch_time_s", "lr"]
    with open(save_dir / f"ccn_{args.n_runs}runs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for run in range(args.n_runs):
            seed = args.seed + run
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model = ResNet18NodeCategorical(var_ranges, grid_size=N,
                                            constraint_mode="none").to(device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
            sched = (torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.epochs, eta_min=args.lr * 0.01)
                if args.scheduler == "cosine" else None)
            print(f"\n--- Run {run + 1}/{args.n_runs} (seed={seed}) ---")
            best, patience = -float("inf"), 0
            for epoch in range(1, args.epochs + 1):
                t = time.time()
                tr = train_fn(model, opt)
                val = eval_fn(model, val_loader)
                if sched is not None:
                    sched.step()
                lr = opt.param_groups[0]["lr"]
                dt = time.time() - t
                writer.writerow([run, epoch, tr, val["val_loss"], val["raw_exact"],
                                 val["raw_hamming"], val["raw_consistent"], val["exact"],
                                 val["hamming"], val["consistent"], val["avg_l2_edit"],
                                 val["argmax_changed_frac"], dt, lr])
                f.flush()
                print(f"  Ep {epoch:3d} | TrLoss {tr:.4f} | VLoss {val['val_loss']:.4f} | "
                      f"Ex/Ham/Con {val['exact']:5.1f}/{val['hamming']:5.1f}/{val['consistent']:5.1f}"
                      f" | Raw {val['raw_exact']:5.1f}/{val['raw_hamming']:5.1f}/"
                      f"{val['raw_consistent']:5.1f} | {dt:.1f}s")
                if val["exact"] > best:
                    best, patience = val["exact"], 0
                    torch.save(model.state_dict(), save_dir / f"best_run{run}.pt")
                else:
                    patience += 1
                    if patience >= args.patience:
                        print(f"  Early stop at epoch {epoch} (best val exact {best:.2f})")
                        break
            model.load_state_dict(torch.load(save_dir / f"best_run{run}.pt",
                                             weights_only=True))
            test = eval_fn(model, test_loader)
            print(f"  Test | Ex/Ham/Con {test['exact']:5.1f}/{test['hamming']:5.1f}/"
                  f"{test['consistent']:5.1f} | Raw {test['raw_exact']:5.1f}/"
                  f"{test['raw_hamming']:5.1f}/{test['raw_consistent']:5.1f}")
            (save_dir / f"test_run{run}.json").write_text(json.dumps(test, indent=2))


if __name__ == "__main__":
    main()
