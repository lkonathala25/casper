import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import WarcraftDataset
from model import ResNet18NodeCategorical, is_valid_path
from node_categories import build_node_categories, node_preds_to_edge_preds


def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("expected boolean")


def _vertex_cost(edge_mask, vw_flat, edge_endpoints, src, sink):
    """Sum vertex_weights over visited vertices (source/sink always counted)."""
    visited = np.unique(np.concatenate([
        edge_endpoints[edge_mask].ravel(),
        np.array([src, sink], dtype=np.int64),
    ]))
    return float(vw_flat[visited].sum())


@torch.no_grad()
def _collect_predictions(model, loader, node_info, node_order, num_edges,
                         device, apply_constraints):
    model.eval()
    pred_all, gt_all = [], []
    for imgs, _, edge_labels in loader:
        imgs = imgs.to(device)
        _, probs, projected, _ = model(imgs, apply_constraints=apply_constraints)
        out = projected if apply_constraints else probs
        edges = node_preds_to_edge_preds(out.cpu(), node_info, node_order, num_edges)
        pred_all.append((edges > 0.5).numpy().astype(np.uint8))
        gt_all.append(edge_labels.numpy().astype(np.uint8))
    return np.concatenate(pred_all, axis=0), np.concatenate(gt_all, axis=0)


def _compute_one(pred_edges, gt_edges, vertex_weights, edges_lookup, N,
                 apply_constraints):
    """Compute path-level + cost-optimality metrics for a single checkpoint's
    predictions. apply_constraints controls how validity is determined:
      - True:  every prediction is treated as a valid path (the projection
               guarantees this), so n_valid = n_total.
      - False: filter to predictions whose raw output is a single source-sink
               path; invalid predictions count as failures for every cost
               threshold.
    """
    n_total = pred_edges.shape[0]

    # Path-level: exact, hamming, consistent.
    exact_pct = float((pred_edges == gt_edges).all(axis=1).mean() * 100)
    hamming_pct = float((pred_edges == gt_edges).mean() * 100)
    consistent_mask = np.array(
        [is_valid_path(pred_edges[i], edges_lookup, N) for i in range(n_total)],
        dtype=bool,
    )
    consistent_pct = float(consistent_mask.mean() * 100)

    # Cost-optimality is meaningful only on valid paths.
    if apply_constraints:
        valid = np.ones(n_total, dtype=bool)
    else:
        valid = consistent_mask
    n_valid = int(valid.sum())

    pred_v = pred_edges[valid].astype(bool)
    gt_v = gt_edges[valid].astype(bool)
    vw_v = vertex_weights[valid].astype(np.float64)

    edges_arr = np.asarray(edges_lookup)
    endpoints = np.stack([
        edges_arr[:, 0] * N + edges_arr[:, 1],
        edges_arr[:, 2] * N + edges_arr[:, 3],
    ], axis=1)
    src = 0
    sink = (N - 1) * N + (N - 1)

    if n_valid > 0:
        ratios = np.empty(n_valid, dtype=np.float64)
        for i in range(n_valid):
            vw_flat = vw_v[i].ravel()
            pred_cost = _vertex_cost(pred_v[i], vw_flat, endpoints, src, sink)
            gt_cost = _vertex_cost(gt_v[i], vw_flat, endpoints, src, sink)
            ratios[i] = pred_cost / max(gt_cost, 1e-12)
    else:
        ratios = np.array([], dtype=np.float64)

    def _full_set_frac(threshold):
        if n_valid == 0:
            return 0.0
        meets = (ratios <= threshold + 1e-6).sum()
        return float(meets) / n_total * 100

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "valid_pct": float(n_valid) / n_total * 100,
        "exact": exact_pct,
        "hamming": hamming_pct,
        "consistent": consistent_pct,
        "ratio_eq_1.0": _full_set_frac(1.0),
        "ratio_le_1.01": _full_set_frac(1.01),
        "ratio_le_1.05": _full_set_frac(1.05),
        "ratio_le_1.10": _full_set_frac(1.10),
        "ratio_mean_on_valid": float(ratios.mean()) if n_valid > 0 else float("nan"),
        "ratio_median_on_valid": float(np.median(ratios)) if n_valid > 0 else float("nan"),
    }


def _resolve_checkpoints(args):
    if args.ckpt:
        ckpts = [Path(args.ckpt)]
    elif args.ckpt_dir:
        root = Path(args.ckpt_dir)
        # Direct hits.
        ckpts = sorted(root.glob("best_run*.pt"))
        # Recursive: <root>/<seed_dir>/<condition>/best_run*.pt etc.
        if not ckpts:
            ckpts = sorted(root.glob("**/best_run*.pt"))
    else:
        raise SystemExit("Pass --ckpt or --ckpt-dir.")
    if not ckpts:
        raise SystemExit(f"No checkpoints found under {args.ckpt or args.ckpt_dir!r}.")
    return ckpts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to a single .pt checkpoint.")
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="Directory containing best_run*.pt (recurses).")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--grid-size", type=int, required=True, choices=[12, 30])
    parser.add_argument("--processing-order", type=str, default="row",
                        choices=["row", "diagonal"])
    parser.add_argument("--apply-constraints", type=_str_to_bool, default=True,
                        help="Apply the constraint layer at inference (default True).")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out_json", type=str, default=None,
                        help="Optional JSON output path; required for downstream "
                             "table aggregation.")
    args = parser.parse_args()

    if args.ckpt and args.ckpt_dir:
        parser.error("Pass --ckpt OR --ckpt-dir, not both.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = args.grid_size

    node_info, var_ranges, node_order, _ = build_node_categories(N)
    data_dir = Path(args.data_dir)

    test_ds = WarcraftDataset(data_dir, "test", node_info, node_order)
    loader = DataLoader(test_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)
    edges_lookup = np.load(data_dir / "edges.npy")
    vertex_weights = np.load(data_dir / "test_vertex_weights.npy")
    print(f"Grid {N}x{N}  test={len(test_ds)}  apply_constraints={args.apply_constraints}")

    ckpts = _resolve_checkpoints(args)
    print(f"Found {len(ckpts)} checkpoint(s).")

    per_ckpt = {}
    for ckpt in ckpts:
        # Tag by relative path from the common root if available, else stem.
        tag = str(ckpt.relative_to(ckpt.parents[2])) if len(ckpt.parents) >= 3 else ckpt.stem
        print(f"\n--- {tag} ---")
        model = ResNet18NodeCategorical(
            var_ranges=var_ranges, grid_size=N,
            processing_order=args.processing_order,
        ).to(device)
        sd = torch.load(ckpt, weights_only=True, map_location=device)
        if isinstance(sd, dict) and "nn_state_dict" in sd:
            sd = sd["nn_state_dict"]
        elif isinstance(sd, dict) and "model_state" in sd:
            sd = sd["model_state"]
        model.load_state_dict(sd)

        pred_edges, gt_edges = _collect_predictions(
            model, loader, node_info, node_order, edges_lookup.shape[0],
            device, apply_constraints=args.apply_constraints,
        )
        m = _compute_one(pred_edges, gt_edges, vertex_weights, edges_lookup, N,
                          apply_constraints=args.apply_constraints)
        per_ckpt[tag] = m
        print(f"  exact={m['exact']:.2f}  hamming={m['hamming']:.2f}  "
              f"consistent={m['consistent']:.2f}  valid={m['valid_pct']:.2f}")
        print(f"  ratio==1.0={m['ratio_eq_1.0']:.2f}  <=1.01={m['ratio_le_1.01']:.2f}  "
              f"<=1.05={m['ratio_le_1.05']:.2f}  <=1.10={m['ratio_le_1.10']:.2f}")

    summary = None
    if len(per_ckpt) > 1:
        keys = ["exact", "hamming", "consistent",
                "ratio_eq_1.0", "ratio_le_1.01", "ratio_le_1.05", "ratio_le_1.10",
                "ratio_mean_on_valid", "ratio_median_on_valid", "valid_pct"]
        summary = {}
        print(f"\n=== Across-checkpoint mean ± std (n={len(per_ckpt)}) ===")
        for k in keys:
            vals = [per_ckpt[t][k] for t in per_ckpt
                    if not (isinstance(per_ckpt[t][k], float)
                            and np.isnan(per_ckpt[t][k]))]
            if not vals:
                continue
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            summary[k] = {"mean": mean, "std": sd, "n": len(vals)}
            print(f"  {k:<24} {mean:8.3f} ± {sd:.3f}  (n={len(vals)})")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "per_ckpt": per_ckpt,
            "summary": summary,
            "ckpt": args.ckpt,
            "ckpt_dir": args.ckpt_dir,
            "data_dir": str(data_dir),
            "grid_size": N,
            "processing_order": args.processing_order,
            "apply_constraints": args.apply_constraints,
        }, indent=2))
        print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
