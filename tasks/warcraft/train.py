import sys as _sys
from pathlib import Path as _Path

# Runnable entry point: put the shared compiler and this task's own directory on
# the path. Sibling task directories are deliberately not added -- they share
# module names (data, model, train, minedit).
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT / "casper", _Path(__file__).resolve().parent):
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import WarcraftDataset
from model import (
    ResNet18NodeCategorical,
    node_categorical_loss_vectorised,
    compute_metrics,
)
from masked_loss import WarcraftMaskedLoss
from node_categories import build_node_categories, node_preds_to_edge_preds


# =============================================================================
# Edge-marginal auxiliary (endpoint-averaged BCE on raw node softmax)
# =============================================================================

def build_edge_marginal_matrix(node_info, node_order, total_dim, num_edges):
    M = torch.zeros(total_dim, num_edges)
    offset = 0
    for node in node_order:
        info = node_info[node]
        for cat_idx, pattern in enumerate(info['edge_patterns']):
            for edge_idx, val in pattern.items():
                if val == 1:
                    M[offset + cat_idx, edge_idx] += 0.5
        offset += info['n_cats']
    return M


def edge_marginal_bce(probs, edge_labels, edge_matrix, eps=1e-6):
    edge_probs = (probs @ edge_matrix).clamp(eps, 1.0 - eps)
    return F.binary_cross_entropy(edge_probs, edge_labels)


def build_edge_endpoint_matrices(node_info, node_order, total_dim,
                                  num_edges, grid_size):
    """Per-endpoint edge-marginal matrices. Same canonicalisation as the 12x12
    version: u = smaller row-major index, v = larger."""
    M_u = torch.zeros(total_dim, num_edges)
    M_v = torch.zeros(total_dim, num_edges)

    n_w = grid_size - 1
    n_right = grid_size * n_w
    edge_endpoints = {}
    for i in range(grid_size):
        for j in range(grid_size):
            if j < grid_size - 1:
                edge_endpoints[i * n_w + j] = ((i, j), (i, j + 1))
            if i < grid_size - 1:
                edge_endpoints[n_right + i * grid_size + j] = ((i, j), (i + 1, j))

    offsets = {}
    off = 0
    for node in node_order:
        offsets[node] = off
        off += node_info[node]['n_cats']

    for e, (u_node, v_node) in edge_endpoints.items():
        for cat_idx, pattern in enumerate(node_info[u_node]['edge_patterns']):
            if pattern.get(e, 0) == 1:
                M_u[offsets[u_node] + cat_idx, e] = 1.0
        for cat_idx, pattern in enumerate(node_info[v_node]['edge_patterns']):
            if pattern.get(e, 0) == 1:
                M_v[offsets[v_node] + cat_idx, e] = 1.0
    return M_u, M_v


def path_length_loss(probs, edge_labels, edge_matrix):
    edge_probs = probs @ edge_matrix
    pred_len = edge_probs.sum(dim=1)
    target_len = edge_labels.sum(dim=1)
    return F.mse_loss(pred_len, target_len)


def pairwise_consistency_loss(probs, M_u, M_v):
    edge_u = probs @ M_u
    edge_v = probs @ M_v
    return F.mse_loss(edge_u, edge_v)


def focal_node_ce(logits, node_labels, var_ranges, gamma=2.0,
                  skip_weight=1.0, path_weight=1.0):
    groups = defaultdict(list)
    offset = 0
    for n_idx, vr in enumerate(var_ranges):
        groups[vr].append((n_idx, offset))
        offset += vr
    total_loss = 0.0
    n_nodes = len(var_ranges)
    use_class_weights = (skip_weight != 1.0) or (path_weight != 1.0)
    for vr, node_list in groups.items():
        logits_stack, targets_stack = [], []
        for n_idx, off in node_list:
            logits_stack.append(logits[:, off:off + vr])
            targets_stack.append(node_labels[:, n_idx])
        logits_cat = torch.cat(logits_stack, dim=0)
        targets_cat = torch.cat(targets_stack, dim=0)
        log_probs = F.log_softmax(logits_cat, dim=1)
        gt_log_probs = log_probs.gather(1, targets_cat.unsqueeze(1)).squeeze(1)
        gt_probs = gt_log_probs.exp()
        focal_weight = (1.0 - gt_probs).pow(gamma)
        if use_class_weights:
            if vr == 2:
                class_w = torch.full_like(targets_cat, path_weight, dtype=logits_cat.dtype)
            else:
                class_w = torch.where(
                    targets_cat == 0,
                    torch.tensor(skip_weight, dtype=logits_cat.dtype, device=logits_cat.device),
                    torch.tensor(path_weight, dtype=logits_cat.dtype, device=logits_cat.device),
                )
            group_loss = (focal_weight * class_w * (-gt_log_probs)).sum() / class_w.sum()
        else:
            group_loss = (-focal_weight * gt_log_probs).mean()
        total_loss = total_loss + group_loss * len(node_list)
    return total_loss / n_nodes


# =============================================================================
# Training loops
# =============================================================================

def train_epoch_ce(model, loader, optimizer, var_ranges, device,
                   skip_weight, path_weight, grad_clip=0.0):
    model.train()
    total_loss = 0.0
    n = 0
    for imgs, node_labels, _ in tqdm(loader, desc="  Train", leave=False):
        imgs = imgs.to(device)
        node_labels = node_labels.to(device)

        logits, _, _, _ = model(imgs, apply_constraints=False)
        loss = node_categorical_loss_vectorised(
            logits, node_labels, var_ranges,
            skip_weight=skip_weight, path_weight=path_weight)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return {"train_loss": total_loss / n,
            "active_set_size_mean": float("nan"),
            "mean_l2_edit_train": float("nan")}


def train_epoch_masked(model, loader, optimizer, masked_loss_fn, var_ranges, device,
                       beta_ce=0.0, grad_clip=0.0,
                       edge_matrix=None, lambda_edge=0.0,
                       M_u=None, M_v=None, lambda_pair=0.0,
                       lambda_len=0.0, gamma_focal=0.0,
                       skip_weight=1.0, path_weight=1.0,
                       lambda_l2_edit=0.0):
    """If lambda_l2_edit > 0, runs the constraint layer on each batch's raw probs
    and adds λ · mean_b ||raw_probs - projected_probs.detach()||² to the loss.
    The projected output is detached so gradient flows only through `probs`,
    pulling the encoder toward the layer's projected distribution. This is a
    self-distillation signal: the encoder is taught to imitate its own
    layer-projected output, so it produces "layer-like" raw predictions even
    when the layer is removed at inference. Unlike MinEdit (which only edits
    cells with arg-max changes), this term penalises ALL deviation, including
    on cells the layer left untouched.
    """
    model.train()
    apply_constraints_in_train = lambda_l2_edit > 0.0
    total_loss = 0.0
    total_ce = 0.0
    total_pos = 0.0
    total_neg = 0.0
    total_l2 = 0.0
    total_active = 0.0
    total_edge = 0.0
    total_pair = 0.0
    total_len = 0.0
    total_l2_edit = 0.0
    n = 0
    for imgs, node_labels, edge_labels in tqdm(loader, desc="  Train", leave=False):
        imgs = imgs.to(device)
        node_labels = node_labels.to(device)
        edge_labels_d = edge_labels.to(device)

        logits, probs, projected, _ = model(
            imgs, apply_constraints=apply_constraints_in_train)
        masked, info = masked_loss_fn(logits, node_labels)

        loss = masked
        if beta_ce > 0.0:
            if gamma_focal > 0.0:
                ce = focal_node_ce(logits, node_labels, var_ranges,
                                   gamma=gamma_focal,
                                   skip_weight=skip_weight, path_weight=path_weight)
            else:
                ce = node_categorical_loss_vectorised(
                    logits, node_labels, var_ranges,
                    skip_weight=skip_weight, path_weight=path_weight)
            loss = loss + beta_ce * ce
            total_ce += ce.item()
        if lambda_edge > 0.0:
            edge_bce = edge_marginal_bce(probs, edge_labels_d, edge_matrix)
            loss = loss + lambda_edge * edge_bce
            total_edge += edge_bce.item()
        if lambda_pair > 0.0 and M_u is not None:
            pair = pairwise_consistency_loss(probs, M_u, M_v)
            loss = loss + lambda_pair * pair
            total_pair += pair.item()
        if lambda_len > 0.0:
            plen = path_length_loss(probs, edge_labels_d, edge_matrix)
            loss = loss + lambda_len * plen
            total_len += plen.item()
        if lambda_l2_edit > 0.0:
            l2_edit = (probs - projected.detach()).pow(2).sum(dim=1).mean()
            loss = loss + lambda_l2_edit * l2_edit
            total_l2_edit += l2_edit.item()

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_pos += info["loss_pos_mean"]
        total_neg += info["loss_neg_mean"]
        total_l2 += info["mean_l2_edit"]
        total_active += info["active_set_size_mean"]
        n += 1

    return {"train_loss": total_loss / n,
            "train_ce": total_ce / n,
            "train_edge": total_edge / n,
            "train_pair": total_pair / n,
            "train_len": total_len / n,
            "train_l2_edit": total_l2_edit / n,
            "loss_pos_mean": total_pos / n,
            "loss_neg_mean": total_neg / n,
            "active_set_size_mean": total_active / n,
            "mean_l2_edit_train": total_l2 / n}


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(model, loader, edges, node_info, node_order, var_ranges, device,
             apply_constraints, num_edges, N,
             skip_weight=1.0, path_weight=1.0):
    model.eval()
    total_loss = 0.0
    n = 0
    total_l2 = 0.0
    total_argmax_changed = 0
    total_cells = 0

    all_raw = []
    all_proj = []
    all_edge_labels = []

    for imgs, node_labels, edge_labels in tqdm(loader, desc="  Eval", leave=False):
        imgs = imgs.to(device)
        node_labels = node_labels.to(device)

        logits, probs, projected, _ = model(imgs, apply_constraints=apply_constraints)
        loss = node_categorical_loss_vectorised(
            logits, node_labels, var_ranges,
            skip_weight=skip_weight, path_weight=path_weight)
        total_loss += loss.item()
        n += 1

        all_raw.append(probs.cpu())
        all_edge_labels.append(edge_labels)

        if apply_constraints:
            all_proj.append(projected.cpu())
            total_l2 += torch.linalg.vector_norm(
                projected - probs, ord=2, dim=1
            ).sum().item()
            offset = 0
            raw_changed = 0
            for vr in var_ranges:
                raw_a = probs[:, offset:offset + vr].argmax(dim=1)
                proj_a = projected[:, offset:offset + vr].argmax(dim=1)
                raw_changed += (raw_a != proj_a).sum().item()
                offset += vr
            total_argmax_changed += raw_changed
            total_cells += probs.shape[0] * len(var_ranges)

    all_raw = torch.cat(all_raw, dim=0)
    all_edge_labels = torch.cat(all_edge_labels, dim=0)

    raw_metrics = compute_metrics(all_raw, all_edge_labels, edges, node_info, node_order,
                                  num_edges, N)

    raw_edge_preds = node_preds_to_edge_preds(all_raw, node_info, node_order, num_edges)
    raw_keys = [v.tobytes() for v in
                (raw_edge_preds > 0.5).cpu().numpy().astype(np.uint8)]
    raw_unique = len(set(raw_keys))
    raw_top1_count = max(Counter(raw_keys).values())
    n_samples = all_raw.shape[0]

    out = {
        "val_loss": total_loss / n,
        "raw_exact": raw_metrics["exact"],
        "raw_hamming": raw_metrics["hamming"],
        "raw_consistent": raw_metrics["consistent"],
        "n_unique_raw": raw_unique,
        "top1_raw_frac": raw_top1_count / n_samples,
    }

    if apply_constraints:
        all_proj = torch.cat(all_proj, dim=0)
        proj_metrics = compute_metrics(all_proj, all_edge_labels, edges, node_info, node_order,
                                       num_edges, N)

        proj_edge_preds = node_preds_to_edge_preds(all_proj, node_info, node_order, num_edges)
        proj_keys = [v.tobytes() for v in
                     (proj_edge_preds > 0.5).cpu().numpy().astype(np.uint8)]
        proj_unique = len(set(proj_keys))
        proj_top1_count = max(Counter(proj_keys).values())

        out.update({
            "proj_exact": proj_metrics["exact"],
            "proj_hamming": proj_metrics["hamming"],
            "proj_consistent": proj_metrics["consistent"],
            "avg_l2_edit": total_l2 / n_samples,
            "argmax_changed_frac": total_argmax_changed / total_cells,
            "n_unique_proj": proj_unique,
            "top1_proj_frac": proj_top1_count / n_samples,
        })
    else:
        out.update({
            "proj_exact": float("nan"),
            "proj_hamming": float("nan"),
            "proj_consistent": float("nan"),
            "avg_l2_edit": float("nan"),
            "argmax_changed_frac": float("nan"),
            "n_unique_proj": float("nan"),
            "top1_proj_frac": float("nan"),
        })

    if apply_constraints:
        out["exact"] = out["proj_exact"]
        out["hamming"] = out["proj_hamming"]
        out["consistent"] = out["proj_consistent"]
    else:
        out["exact"] = out["raw_exact"]
        out["hamming"] = out["raw_hamming"]
        out["consistent"] = out["raw_consistent"]

    return out


# =============================================================================
# Condition runner
# =============================================================================

def run_condition(condition, args, device, train_loader, val_loader, test_loader,
                  edges, node_info, node_order, var_ranges, num_edges, N,
                  edge_matrix=None, M_u=None, M_v=None, steps_artifact=None):
    # `crl_in_training_eval_off` keeps CaRL active during training (via masked
    # loss + optional λ_l2_edit self-distillation) but disables the constraint
    # layer at val/test, so the per-epoch metrics track raw-backbone behaviour.
    # Used to probe whether the backbone has internalised the constraint.
    apply_constraints_at_eval = condition in ("crl_inference_only", "crl_in_training")
    use_masked_loss = condition in ("crl_in_training", "crl_in_training_eval_off")

    save_dir = Path(args.save_dir) / condition
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"{condition}_{args.n_runs}runs.csv"

    print(f"\n{'=' * 70}")
    print(f"CONDITION: {condition}")
    print(f"  epochs={args.epochs}, runs={args.n_runs}, lr={args.lr}, grid={N}x{N}")
    print(f"  train-time CRL (WarcraftMaskedLoss): {use_masked_loss} "
          f"{'(λ_neg=' + str(args.lambda_neg) + ')' if use_masked_loss else ''} "
          f"| eval-time CRL: {apply_constraints_at_eval}")
    if use_masked_loss:
        print(f"  aux: edge={args.lambda_edge} pair={args.lambda_pair} "
              f"len={args.lambda_len} beta_ce={args.beta_ce} focal_gamma={args.gamma_focal}")
    if args.skip_weight != 1.0 or args.path_weight != 1.0:
        print(f"  CE weights: skip={args.skip_weight} path={args.path_weight}")
    print(f"  CSV: {csv_path}")
    print(f"{'=' * 70}")

    header = [
        "condition", "run_id", "epoch", "train_loss", "val_loss",
        "exact", "hamming", "consistent",
        "raw_exact", "raw_hamming", "raw_consistent",
        "proj_exact", "proj_hamming", "proj_consistent",
        "avg_l2_edit", "argmax_changed_frac",
        "n_unique_raw", "top1_raw_frac",
        "n_unique_proj", "top1_proj_frac",
        "active_set_size_mean", "mean_l2_edit_train",
        "lr", "epoch_time_s",
        "gpu_mem_mb", "gpu_mem_peak_mb",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        f.flush()

        for run in range(args.n_runs):
            seed = args.seed + run
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model = ResNet18NodeCategorical(
                var_ranges=var_ranges, grid_size=N,
                processing_order=args.processing_order,
                constraint_mode=args.constraint_mode,
                alphabet=args.alphabet,
                artifact=args.artifact,
                max_rounds=args.max_rounds,
                steps_artifact=steps_artifact,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                         weight_decay=args.weight_decay)
            masked_loss_fn = (WarcraftMaskedLoss(var_ranges, lambda_neg=args.lambda_neg).to(device)
                              if use_masked_loss else None)
            if args.scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
                )
            elif args.scheduler == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=5
                )
            else:
                scheduler = None

            print(f"\n--- Run {run + 1}/{args.n_runs} (seed={seed}) ---")

            best_val_exact = -float("inf")
            patience_counter = 0

            for epoch in range(1, args.epochs + 1):
                t0 = time.time()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)

                if use_masked_loss:
                    train_info = train_epoch_masked(
                        model, train_loader, optimizer, masked_loss_fn,
                        var_ranges, device, beta_ce=args.beta_ce,
                        grad_clip=args.grad_clip,
                        edge_matrix=edge_matrix, lambda_edge=args.lambda_edge,
                        M_u=M_u, M_v=M_v, lambda_pair=args.lambda_pair,
                        lambda_len=args.lambda_len, gamma_focal=args.gamma_focal,
                        skip_weight=args.skip_weight, path_weight=args.path_weight,
                        lambda_l2_edit=args.lambda_l2_edit)
                else:
                    train_info = train_epoch_ce(
                        model, train_loader, optimizer, var_ranges, device,
                        skip_weight=args.skip_weight, path_weight=args.path_weight,
                        grad_clip=args.grad_clip)

                val = evaluate(model, val_loader, edges, node_info, node_order,
                               var_ranges, device,
                               apply_constraints=apply_constraints_at_eval,
                               num_edges=num_edges, N=N,
                               skip_weight=args.skip_weight, path_weight=args.path_weight)

                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val["val_loss"])
                elif scheduler is not None:
                    scheduler.step()
                lr = optimizer.param_groups[0]["lr"]
                dt = time.time() - t0
                if device.type == "cuda":
                    gpu_mem_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
                    gpu_mem_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                else:
                    gpu_mem_mb = float("nan")
                    gpu_mem_peak_mb = float("nan")

                writer.writerow([
                    condition, run, epoch,
                    train_info["train_loss"], val["val_loss"],
                    val["exact"], val["hamming"], val["consistent"],
                    val["raw_exact"], val["raw_hamming"], val["raw_consistent"],
                    val["proj_exact"], val["proj_hamming"], val["proj_consistent"],
                    val["avg_l2_edit"], val["argmax_changed_frac"],
                    val["n_unique_raw"], val["top1_raw_frac"],
                    val["n_unique_proj"], val["top1_proj_frac"],
                    train_info["active_set_size_mean"], train_info["mean_l2_edit_train"],
                    lr, dt,
                    gpu_mem_mb, gpu_mem_peak_mb,
                ])
                f.flush()

                extra = ""
                if apply_constraints_at_eval:
                    extra = (f" | Raw Ex/Ham/Con: "
                             f"{val['raw_exact']:5.1f}/{val['raw_hamming']:5.1f}/{val['raw_consistent']:5.1f}"
                             f" | L2: {val['avg_l2_edit']:.4f}"
                             f" | ArgΔ: {val['argmax_changed_frac']:.3f}"
                             f" | Unq R/P: {val['n_unique_raw']:4d}/{val['n_unique_proj']:4d}"
                             f" (top-1 P: {val['top1_proj_frac']*100:5.1f}%)")
                else:
                    extra = (f" | Unq R: {val['n_unique_raw']:4d}"
                             f" (top-1 R: {val['top1_raw_frac']*100:5.1f}%)")
                if use_masked_loss:
                    extra += (f" | Pos: {train_info['loss_pos_mean']:.4f}"
                              f" | Neg: {train_info['loss_neg_mean']:.4f}")
                    if args.beta_ce > 0.0:
                        extra += (f" | CE(β={args.beta_ce:g}): "
                                  f"{train_info['train_ce']:.4f}")
                    if args.lambda_edge > 0.0:
                        extra += (f" | EdgeBCE(λ={args.lambda_edge:g}): "
                                  f"{train_info['train_edge']:.4f}")
                    if args.lambda_pair > 0.0:
                        extra += (f" | Pair(λ={args.lambda_pair:g}): "
                                  f"{train_info['train_pair']:.4f}")
                    if args.lambda_len > 0.0:
                        extra += (f" | Len(λ={args.lambda_len:g}): "
                                  f"{train_info['train_len']:.4f}")
                    if args.lambda_l2_edit > 0.0:
                        extra += (f" | L2Edit(λ={args.lambda_l2_edit:g}): "
                                  f"{train_info['train_l2_edit']:.4f}")

                print(f"  Ep {epoch:3d} | TrLoss: {train_info['train_loss']:.4f} | "
                      f"VLoss: {val['val_loss']:.4f} | "
                      f"Ex/Ham/Con: "
                      f"{val['exact']:5.1f}/{val['hamming']:5.1f}/{val['consistent']:5.1f}"
                      f"{extra} | LR: {lr:.2e} | {dt:.1f}s")

                # Select on what is reported on test: the projected exact match
                # when the layer is applied at inference, the raw one otherwise.
                select_metric = (val["proj_exact"]
                                 if condition == "crl_in_training" and apply_constraints_at_eval
                                 else val["exact"])
                if select_metric > best_val_exact:
                    best_val_exact = select_metric
                    patience_counter = 0
                    torch.save(model.state_dict(),
                               save_dir / f"best_run{run}.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f"  Early stopping at epoch {epoch} "
                              f"(best select={best_val_exact:.2f})")
                        break

            model.load_state_dict(torch.load(
                save_dir / f"best_run{run}.pt", weights_only=True))
            test = evaluate(model, test_loader, edges, node_info, node_order,
                            var_ranges, device,
                            apply_constraints=apply_constraints_at_eval,
                            num_edges=num_edges, N=N,
                            skip_weight=args.skip_weight, path_weight=args.path_weight)
            print(f"  Test | Ex/Ham/Con: "
                  f"{test['exact']:5.1f}/{test['hamming']:5.1f}/{test['consistent']:5.1f}"
                  + (f" | Raw: "
                     f"{test['raw_exact']:5.1f}/{test['raw_hamming']:5.1f}/{test['raw_consistent']:5.1f}"
                     f" | L2: {test['avg_l2_edit']:.4f}"
                     if apply_constraints_at_eval else ""))

            with open(save_dir / f"test_run{run}.json", "w") as tf:
                json.dump(test, tf, indent=2)

    print(f"\nResults → {csv_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results/warcraft_ablation_nd")
    parser.add_argument("--grid-size", type=int, required=True,
                        help="Grid size N (e.g. 12 or 30). No default; must match "
                             "the preprocessed data at --data_dir.")
    parser.add_argument("--condition", type=str, default="all",
                        choices=["all", "no_crl", "crl_inference_only",
                                 "crl_in_training", "crl_in_training_eval_off"])
    parser.add_argument("--n-runs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-weight", type=float, default=1.0,
                        help="CE weight on skip-label cells (n_cats>2, target==0). "
                             "1.0 = unweighted. At 30x30, where ~98%% of cells are "
                             "skip, setting path_weight > 1.0 counteracts this.")
    parser.add_argument("--path-weight", type=float, default=1.0,
                        help="CE weight on on-path cells (routing categories, "
                             "and all source/sink cells). Paired with --skip-weight.")
    parser.add_argument("--lambda-neg", type=float, default=1.0)
    parser.add_argument("--beta-ce", type=float, default=0.0)
    parser.add_argument("--lambda-edge", type=float, default=0.0)
    parser.add_argument("--lambda-pair", type=float, default=0.0)
    parser.add_argument("--lambda-len", type=float, default=0.0)
    parser.add_argument("--gamma-focal", type=float, default=0.0)
    parser.add_argument("--lambda-l2-edit", type=float, default=0.0,
                        help="Weight on the L2 self-distillation term: λ · "
                             "mean_b ||raw_probs - projected_probs.detach()||². "
                             "Forward runs with apply_constraints=True so the "
                             "constraint layer's output is available; gradient "
                             "flows only through `probs`, pulling the encoder "
                             "toward the layer-projected distribution. Adds "
                             "modest per-epoch cost (one extra projection pass). "
                             "0 disables. Try 0.5-2.0.")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau", "none"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processing-order", type=str, default="row",
                        choices=["row", "diagonal"],
                        help="Node visit order in WarcraftConstraintLayer.forward. "
                             "'row' is the original row-major sweep; 'diagonal' "
                             "sweeps anti-diagonals (i+j ascending) with top-down "
                             "tiebreak. Both are valid topological orders.")
    parser.add_argument("--constraint-mode", type=str, default="steps",
                        choices=["none", "monotone", "lazy_exact", "steps"],
                        help="Projection layer. 'none' builds no layer (only with "
                             "--condition no_crl). 'monotone' is the original "
                             "down/right flow sweep, retained only for "
                             "reproducing the superseded results; it cannot "
                             "represent non-monotone paths and must be asked "
                             "for explicitly. "
                             "'lazy_exact' is the compiled degree clauses plus "
                             "ring elimination; 'steps' is the sequential "
                             "min-edit projection over the trimmed steps "
                             "encoding. Must match --alphabet.")
    parser.add_argument("--artifact", type=str, default=None,
                        help="Constraint artifact path for --constraint-mode "
                             "lazy_exact; ignored by the other modes.")
    parser.add_argument("--steps-artifact", type=str, default=None,
                        help="Compiled steps artifact checked at startup under "
                             "--constraint-mode steps. Defaults to "
                             "constraints/warcraft/steps_artifacts/"
                             "steps_<N>_trimmed_compiled.json.")
    parser.add_argument("--skip-steps-gate", action="store_true",
                        help="Skip the cross-check of the loaded artifact "
                             "against the steps encoding. The artifact is still "
                             "loaded and enforced; only the check is dropped.")
    parser.add_argument("--max-rounds", type=int, default=20,
                        help="Ring-repair round cap for --constraint-mode "
                             "lazy_exact. The steps layer has no repair loop: "
                             "loop erasure is a single deterministic pass.")
    parser.add_argument("--alphabet", type=str, default="full",
                        choices=["monotone", "full"],
                        help="Node category alphabet. 'monotone' represents only "
                             "down/right traversal and rejects non-monotone "
                             "ground truth; 'full' represents every 4-connected "
                             "simple path.")
    args = parser.parse_args()
    if args.constraint_mode == "none" and args.condition != "no_crl":
        parser.error("--constraint-mode none has no layer; use it only with "
                     "--condition no_crl")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    N = args.grid_size
    node_info, var_ranges, node_order, total_dim = build_node_categories(
        N, alphabet=args.alphabet)
    print(f"Grid: {N}x{N} | Nodes: {len(node_order)}, total_dim: {total_dim}")

    data_dir = Path(args.data_dir)
    # Startup gate: WarcraftDataset converts edge labels to node labels through
    # the fail-loud converter, so an unrepresentable pattern raises here rather
    # than silently corrupting a training target. Report it instead of relying
    # on the absence of an exception.
    train_ds = WarcraftDataset(data_dir, "train", node_info, node_order)
    val_ds = WarcraftDataset(data_dir, "val", node_info, node_order)
    test_ds = WarcraftDataset(data_dir, "test", node_info, node_order)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Label conversion gate (alphabet={args.alphabet!r}): clean over "
          f"{len(train_ds) + len(val_ds) + len(test_ds)} instances, "
          f"0 UnrepresentablePatternError")

    if args.alphabet == "monotone" or args.constraint_mode == "monotone":
        print("!" * 72)
        print("WARNING: monotone selected. The monotone alphabet represents only")
        print("down/right traversal and the monotone sweep is a single forward pass")
        print("valid only on the R/D DAG. This is the superseded encoding and does")
        print("not represent every 4-connected path. Use --alphabet full")
        print("--constraint-mode steps for the corrected setting.")
        print("!" * 72)

    # The steps layer walks the compiled clauses themselves, so the artifact is
    # required, not optional. --skip-steps-gate drops only the independent
    # cross-check against the generator, never the load.
    steps_artifact = None
    if args.constraint_mode == "steps":
        steps_artifact = Path(args.steps_artifact) if args.steps_artifact else (
            _ROOT / "constraints" / "warcraft" / "steps_artifacts"
            / f"steps_{N}_trimmed_compiled.json"
        )
        if not steps_artifact.exists():
            raise FileNotFoundError(
                f"steps mode enforces the compiled artifact {steps_artifact}, "
                f"which does not exist. Build it with "
                f"tasks/warcraft/build_steps_artifacts.py --size {N} then "
                f"tasks/warcraft/compile_steps_encoding.py on the emitted DSL."
            )
        print(f"Steps constraints loaded from {steps_artifact.name}")

        if not args.skip_steps_gate:
            from steps_walk import StepsWalkDecoder
            t_gate = time.time()
            gen = StepsWalkDecoder(N)
            afx = StepsWalkDecoder(N, compiled_artifact=steps_artifact)
            if gen.enc.domains != afx.enc.domains:
                raise RuntimeError(
                    f"{steps_artifact} declares different per-tick domains than "
                    f"the steps encoding for a {N}x{N} grid; refusing to train."
                )
            bad = [
                (i, u)
                for i in range(1, gen.enc.horizon + 1)
                for u in gen.enc.domains[i - 1]
                if tuple(sorted(gen.feasible(i, u)))
                != tuple(sorted(afx.feasible(i, u)))
            ]
            if bad:
                raise RuntimeError(
                    f"{len(bad)} feasible sets derived from {steps_artifact.name} "
                    f"disagree with the steps encoding (first: tick {bad[0][0]}, "
                    f"cell {bad[0][1]}); refusing to train."
                )
            print(f"Compiled-artifact gate: all "
                  f"{sum(len(d) for d in gen.enc.domains[:-1])} feasible sets "
                  f"agree with the encoding ({time.time() - t_gate:.1f}s)")

    dropped = {k: getattr(args, k) for k in
               ("beta_ce", "gamma_focal", "lambda_edge", "lambda_pair", "lambda_len")
               if getattr(args, k) == 0.0}
    if dropped:
        print(f"Auxiliary losses off: {', '.join(sorted(dropped))}")
    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    (save_root / "config.json").write_text(json.dumps(
        {"args": vars(args), "total_dim": total_dim,
         "auxiliary_losses_off": sorted(dropped)}, indent=2, default=str))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    edges = np.load(data_dir / "edges.npy")
    num_edges = edges.shape[0]
    assert num_edges == 2 * N * (N - 1), \
        f"edges.npy has {num_edges} rows but grid_size={N} implies {2*N*(N-1)}"

    edge_matrix = build_edge_marginal_matrix(
        node_info, node_order, total_dim, num_edges=num_edges).to(device)
    M_u, M_v = build_edge_endpoint_matrices(
        node_info, node_order, total_dim,
        num_edges=num_edges, grid_size=N)
    M_u = M_u.to(device)
    M_v = M_v.to(device)

    conditions = (["no_crl", "crl_inference_only", "crl_in_training"]
                  if args.condition == "all" else [args.condition])

    for cond in conditions:
        run_condition(cond, args, device, train_loader, val_loader, test_loader,
                      edges, node_info, node_order, var_ranges, num_edges, N,
                      edge_matrix=edge_matrix, M_u=M_u, M_v=M_v,
                      steps_artifact=steps_artifact)


if __name__ == "__main__":
    main()
