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
from torch.utils.data import DataLoader
from tqdm import tqdm

from casper_loss import (CASPER_MODES, VariableLayout, casper_loss, layer_at_inference,
                         layer_in_training, raw_nll, resolve_l2_penalty)
from data import WarcraftDataset
from resource_log import RESOURCE_FIELDS, EpochResources
from model import ResNet18NodeCategorical, compute_metrics
from node_categories import build_node_categories, node_preds_to_edge_preds


def class_weights(node_labels, binary_nodes, skip_weight, path_weight):
    """(B, n_nodes) per-node loss weights, or None when both weights are 1.

    Binary nodes (source/sink) always take path_weight; the others take
    skip_weight when their target is skip (category 0) and path_weight otherwise.
    """
    if skip_weight == 1.0 and path_weight == 1.0:
        return None
    on_path = binary_nodes.unsqueeze(0) | (node_labels != 0)
    return torch.where(on_path,
                       torch.full_like(node_labels, path_weight, dtype=torch.float),
                       torch.full_like(node_labels, skip_weight, dtype=torch.float))


def train_epoch(model, loader, optimizer, layout, binary_nodes, device, use_layer, l2_penalty,
                skip_weight=1.0, path_weight=1.0):
    """One epoch of the shared CASPER objective (casper_loss.casper_loss).

    With use_layer the compiled layer runs in the forward pass, the task loss is
    on its projected output (gradients flow through the projection) and
    l2_penalty weights ½‖z̃ − ŷ‖²; without it the task loss is on the raw logits.
    grad_norm is the total gradient norm; gradients are not clipped, as in MNIST.
    """
    model.train()
    sums, n = defaultdict(float), 0
    for imgs, node_labels, _ in tqdm(loader, desc="  Train", leave=False):
        imgs = imgs.to(device)
        node_labels = node_labels.to(device)

        logits, probs, projected, _ = model(imgs, apply_constraints=use_layer)
        weights = class_weights(node_labels, binary_nodes, skip_weight, path_weight)
        loss, parts = casper_loss(layout, logits, probs, projected if use_layer else None,
                                  node_labels, l2_penalty, weights)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.sqrt(sum((q.grad ** 2).sum() for q in model.parameters()
                                   if q.grad is not None))
        optimizer.step()

        sums["train_loss"] += loss.item()
        sums["grad_norm"] += float(grad_norm)
        for k, v in parts.items():
            sums[f"train_{k}"] += v
        n += 1

    return {k: v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate(model, loader, edges, node_info, node_order, var_ranges, device,
             apply_constraints, num_edges, N, layout, binary_nodes,
             skip_weight=1.0, path_weight=1.0):
    """Raw and (with apply_constraints) projected metrics; val_loss is the raw-logit task loss."""
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
        loss = raw_nll(layout, logits, node_labels,
                       class_weights(node_labels, binary_nodes, skip_weight, path_weight))
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


def run_mode(args, device, train_loader, val_loader, test_loader,
             edges, node_info, node_order, var_ranges, num_edges, N, steps_artifact=None):
    """Train --n-runs seeds under --casper-mode and score each one's validation-selected checkpoint on test.

    The checkpoint is selected on validation exact match as reported at test:
    projected when the layer runs at inference, raw otherwise.
    """
    mode = args.casper_mode
    use_layer_in_training = layer_in_training(mode)
    apply_constraints_at_eval = layer_at_inference(mode)
    layout = VariableLayout(var_ranges)
    binary_nodes = torch.tensor([vr == 2 for vr in var_ranges], device=device)

    save_dir = Path(args.save_dir) / mode
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"{mode}_{args.n_runs}runs.csv"

    print(f"\n{'=' * 70}")
    print(f"CASPER MODE: {mode}  (layer in training: {use_layer_in_training}, "
          f"at inference: {apply_constraints_at_eval}, l2_penalty: {args.l2_penalty:g})")
    print(f"  epochs={args.epochs}, runs={args.n_runs}, lr={args.lr}, grid={N}x{N}")
    if args.skip_weight != 1.0 or args.path_weight != 1.0:
        print(f"  loss weights: skip={args.skip_weight} path={args.path_weight}")
    print(f"  CSV: {csv_path}")
    print(f"{'=' * 70}")

    header = [
        "casper_mode", "l2_penalty", "run_id", "epoch",
        "train_loss", "train_task_loss", "train_l2_penalty_term", "train_l2_edit", "grad_norm",
        "val_loss", "exact", "hamming", "consistent",
        "raw_exact", "raw_hamming", "raw_consistent",
        "proj_exact", "proj_hamming", "proj_consistent",
        "avg_l2_edit", "argmax_changed_frac",
        "n_unique_raw", "top1_raw_frac",
        "n_unique_proj", "top1_proj_frac",
        "lr", *RESOURCE_FIELDS,
    ]
    eval_kwargs = dict(apply_constraints=apply_constraints_at_eval, num_edges=num_edges, N=N,
                       layout=layout, binary_nodes=binary_nodes,
                       skip_weight=args.skip_weight, path_weight=args.path_weight)

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
            best_raw_val_exact = -float("inf")
            patience_counter = 0
            res = EpochResources(device)

            for epoch in range(1, args.epochs + 1):
                res.start()
                with res.phase("train"):
                    train_info = train_epoch(
                        model, train_loader, optimizer, layout, binary_nodes, device,
                        use_layer=use_layer_in_training, l2_penalty=args.l2_penalty,
                        skip_weight=args.skip_weight, path_weight=args.path_weight)
                with res.phase("eval"):
                    val = evaluate(model, val_loader, edges, node_info, node_order,
                                   var_ranges, device, **eval_kwargs)

                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val["val_loss"])
                elif scheduler is not None:
                    scheduler.step()
                lr = optimizer.param_groups[0]["lr"]
                cost = res.row(len(train_loader.dataset))
                dt = cost["epoch_time_s"]

                writer.writerow([
                    mode, args.l2_penalty, run, epoch,
                    train_info["train_loss"], train_info["train_task_loss"],
                    train_info["train_l2_penalty_term"], train_info["train_l2_edit"],
                    train_info["grad_norm"],
                    val["val_loss"], val["exact"], val["hamming"], val["consistent"],
                    val["raw_exact"], val["raw_hamming"], val["raw_consistent"],
                    val["proj_exact"], val["proj_hamming"], val["proj_consistent"],
                    val["avg_l2_edit"], val["argmax_changed_frac"],
                    val["n_unique_raw"], val["top1_raw_frac"],
                    val["n_unique_proj"], val["top1_proj_frac"],
                    lr, *(cost[k] for k in RESOURCE_FIELDS),
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
                if use_layer_in_training:
                    extra += (f" | Task: {train_info['train_task_loss']:.4f}"
                              f" | L2Pen(λ={args.l2_penalty:g}): "
                              f"{train_info['train_l2_penalty_term']:.4f}")

                print(f"  Ep {epoch:3d} | TrLoss: {train_info['train_loss']:.4f} | "
                      f"VLoss: {val['val_loss']:.4f} | "
                      f"Ex/Ham/Con: "
                      f"{val['exact']:5.1f}/{val['hamming']:5.1f}/{val['consistent']:5.1f}"
                      f"{extra} | |g|: {train_info['grad_norm']:.2f} | LR: {lr:.2e} | {dt:.1f}s")

                # Train-Only is this run read without the layer at test, so it gets its
                # own checkpoint selected on raw validation exact match.
                if mode == "train_infer" and val["raw_exact"] > best_raw_val_exact:
                    best_raw_val_exact = val["raw_exact"]
                    torch.save(model.state_dict(), save_dir / f"best_raw_run{run}.pt")
                if val["exact"] > best_val_exact:
                    best_val_exact = val["exact"]
                    patience_counter = 0
                    torch.save(model.state_dict(),
                               save_dir / f"best_run{run}.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience and epoch >= args.min_epochs:
                        print(f"  Early stopping at epoch {epoch} "
                              f"(best select={best_val_exact:.2f})")
                        break

            model.load_state_dict(torch.load(
                save_dir / f"best_run{run}.pt", weights_only=True))
            test = evaluate(model, test_loader, edges, node_info, node_order,
                            var_ranges, device, **eval_kwargs)
            print(f"  Test | Ex/Ham/Con: "
                  f"{test['exact']:5.1f}/{test['hamming']:5.1f}/{test['consistent']:5.1f}"
                  + (f" | Raw: "
                     f"{test['raw_exact']:5.1f}/{test['raw_hamming']:5.1f}/{test['raw_consistent']:5.1f}"
                     f" | L2: {test['avg_l2_edit']:.4f}"
                     if apply_constraints_at_eval else ""))

            with open(save_dir / f"test_run{run}.json", "w") as tf:
                json.dump(test, tf, indent=2)

    print(f"\nResults → {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results/warcraft_ablation_nd")
    parser.add_argument("--grid-size", type=int, required=True,
                        help="Grid size N (e.g. 12 or 30). No default; must match "
                             "the preprocessed data at --data_dir.")
    parser.add_argument("--casper-mode", type=str, required=True, choices=CASPER_MODES,
                        help="Where the layer runs. 'train_infer': in training (task loss "
                             "on its projected output) and at inference; 'train_only': "
                             "training only; 'infer_only': raw-logit task loss, layer "
                             "applied only at inference; 'none': never (the ResNet baseline).")
    parser.add_argument("--l2-penalty", type=float, default=None,
                        help="Weight of ½‖z̃ − ŷ‖² added to the task loss. Required with "
                             "train_infer/train_only (1.0 = full CASPER, 0 = the "
                             "w/o-l2-penalty ablation); rejected otherwise.")
    parser.add_argument("--n-runs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=0,
                        help="Early stopping cannot end a run before this epoch.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-weight", type=float, default=1.0,
                        help="CE weight on skip-label cells (n_cats>2, target==0). "
                             "1.0 = unweighted. At 30x30, where ~98%% of cells are "
                             "skip, setting path_weight > 1.0 counteracts this.")
    parser.add_argument("--path-weight", type=float, default=1.0,
                        help="CE weight on on-path cells (routing categories, "
                             "and all source/sink cells). Paired with --skip-weight.")
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
                             "--casper-mode none). 'monotone' is the original "
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
    try:
        args.l2_penalty = resolve_l2_penalty(args.casper_mode, args.l2_penalty)
    except ValueError as err:
        parser.error(str(err))
    if (args.constraint_mode == "none") != (args.casper_mode == "none"):
        parser.error("--constraint-mode none builds no layer: use it exactly when "
                     "--casper-mode is none")

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

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    (save_root / "config.json").write_text(json.dumps(
        {"args": vars(args), "total_dim": total_dim}, indent=2, default=str))

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

    run_mode(args, device, train_loader, val_loader, test_loader,
             edges, node_info, node_order, var_ranges, num_edges, N,
             steps_artifact=steps_artifact)


if __name__ == "__main__":
    main()
