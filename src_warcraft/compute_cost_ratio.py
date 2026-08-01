import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import WarcraftDataset
from model import ResNet18NodeCategorical
from node_categories import build_node_categories, node_preds_to_edge_preds


# ---------------------------------------------------------------------------
# Path-validity check
# ---------------------------------------------------------------------------

def _is_valid_src_sink_path(pred_mask, edge_endpoints, src_flat, sink_flat):
    """True iff the selected edges form a single simple src→sink path.

    Conditions: src/sink each have degree 1, every other involved vertex has
    degree 2, the involved-vertex subgraph is connected, and #edges = #verts − 1.
    """
    if not pred_mask.any():
        return False
    sel = edge_endpoints[pred_mask]
    verts, degrees = np.unique(sel.ravel(), return_counts=True)
    if src_flat not in verts or sink_flat not in verts:
        return False
    if sel.shape[0] != verts.size - 1:
        return False
    deg = dict(zip(verts.tolist(), degrees.tolist()))
    if deg[src_flat] != 1 or deg[sink_flat] != 1:
        return False
    for v, d in deg.items():
        if v == src_flat or v == sink_flat:
            continue
        if d != 2:
            return False
    adj = {int(v): [] for v in verts.tolist()}
    for a, b in sel.tolist():
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    seen = set()
    stack = [int(src_flat)]
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        stack.extend(adj[v])
    return seen == set(int(v) for v in verts.tolist())


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------

def compute_cost_metrics(pred_edges, gt_edges, vertex_weights, edges_lookup, grid_size):
    """Compute per-sample cost ratios and aggregate stats.

    Args:
        pred_edges:     (n_samples, n_edges) binary (np.uint8/bool/int).
        gt_edges:       (n_samples, n_edges) binary.
        vertex_weights: (n_samples, grid_size, grid_size) float, per-vertex cost.
        edges_lookup:   (n_edges, 4) int, each row (r1, c1, r2, c2).
        grid_size:      int (e.g. 30).

    Returns:
        dict with per-sample arrays and aggregate stats.
    """
    pred = np.asarray(pred_edges).astype(bool)
    gt = np.asarray(gt_edges).astype(bool)
    vw = np.asarray(vertex_weights).astype(np.float64)
    edges = np.asarray(edges_lookup).astype(np.int64)
    n_samples, n_edges = pred.shape
    N = grid_size

    # (n_edges, 2) flat node indices for each edge endpoint.
    endpoint_a = edges[:, 0] * N + edges[:, 1]
    endpoint_b = edges[:, 2] * N + edges[:, 3]
    edge_endpoints = np.stack([endpoint_a, endpoint_b], axis=1)  # (n_edges, 2)

    src_flat = 0
    sink_flat = (N - 1) * N + (N - 1)

    pred_cost = np.zeros(n_samples, dtype=np.float64)
    gt_cost = np.zeros(n_samples, dtype=np.float64)
    n_pred_visited = np.zeros(n_samples, dtype=np.int64)
    n_gt_visited = np.zeros(n_samples, dtype=np.int64)
    is_valid_path = np.zeros(n_samples, dtype=bool)

    for i in range(n_samples):
        flat_w = vw[i].ravel()  # (N*N,)

        # Predicted visited set (used for cost when prediction is a valid path)
        active = edge_endpoints[pred[i]]                  # (k_pred, 2)
        visited = np.unique(np.concatenate([
            active.ravel(), [src_flat, sink_flat]
        ]))
        pred_cost[i] = flat_w[visited].sum()
        n_pred_visited[i] = visited.size
        is_valid_path[i] = _is_valid_src_sink_path(
            pred[i], edge_endpoints, src_flat, sink_flat
        )

        # GT visited set
        active = edge_endpoints[gt[i]]
        visited = np.unique(np.concatenate([
            active.ravel(), [src_flat, sink_flat]
        ]))
        gt_cost[i] = flat_w[visited].sum()
        n_gt_visited[i] = visited.size

    # Guard against zero GT cost (shouldn't happen but be defensive).
    gt_cost_safe = np.where(gt_cost > 0, gt_cost, 1.0)
    cost_ratio = pred_cost / gt_cost_safe
    cost_ratio[gt_cost <= 0] = np.nan
    # Inconsistent predictions: assign +inf so they fail every threshold
    # comparison (drop from numerator) while still counting toward the
    # denominator via `n_samples`.
    cost_ratio[~is_valid_path] = np.inf

    # Exact-match (per-edge) for slicing failing samples.
    exact_match = (pred == gt).all(axis=1)
    failing = ~exact_match

    # Aggregate stats: restrict to finite values so inf/nan from invalid
    # predictions or zero-GT samples don't poison mean/median.
    def _stats(arr, mask=None):
        a = arr if mask is None else arr[mask]
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {"mean": float("nan"), "median": float("nan"), "n": 0}
        return {"mean": float(np.mean(a)),
                "median": float(np.median(a)),
                "n": int(a.size)}

    # Tolerance for "cost-optimal" — float arithmetic on float16-derived values.
    EQ_TOL = 1e-6

    frac = lambda mask: float(mask.mean())
    fractions = {
        "ratio_eq_1.0":    frac(np.abs(cost_ratio - 1.0) < EQ_TOL),
        "ratio_le_1.01":   frac(cost_ratio <= 1.01 + EQ_TOL),
        "ratio_le_1.05":   frac(cost_ratio <= 1.05 + EQ_TOL),
        "ratio_le_1.10":   frac(cost_ratio <= 1.10 + EQ_TOL),
    }

    # Distribution histogram.
    bin_edges = [1.0, 1.01, 1.05, 1.10, 1.25, 1.50, 2.0, 5.0, np.inf]
    bin_labels = [
        "[1.00, 1.01)", "[1.01, 1.05)", "[1.05, 1.10)", "[1.10, 1.25)",
        "[1.25, 1.50)", "[1.50, 2.00)", "[2.00, 5.00)", "[5.00, inf)",
    ]
    # < 1.0 bin handled separately so left-edge of 1.0 is unambiguous.
    finite_ratio = np.isfinite(cost_ratio)
    below_one_count = int(np.sum((cost_ratio < 1.0 - EQ_TOL) & finite_ratio))
    hist_counts = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (cost_ratio >= lo - EQ_TOL) & (cost_ratio < hi - EQ_TOL) & finite_ratio
        hist_counts.append(int(mask.sum()))
    histogram = {"<1.0 (degenerate)": below_one_count}
    for lbl, c in zip(bin_labels, hist_counts):
        histogram[lbl] = c
    histogram["invalid (no path)"] = int((~is_valid_path).sum())

    n_invalid = int((~is_valid_path).sum())
    return {
        "per_sample": {
            "pred_cost": pred_cost.tolist(),
            "gt_cost": gt_cost.tolist(),
            "cost_ratio": cost_ratio.tolist(),
            "exact_match": exact_match.tolist(),
            "is_valid_path": is_valid_path.tolist(),
            "n_pred_visited": n_pred_visited.tolist(),
            "n_gt_visited": n_gt_visited.tolist(),
        },
        "aggregate": {
            "n_total": int(n_samples),
            "n_failing": int(failing.sum()),
            "n_invalid": n_invalid,
            "all_samples_cost_ratio": _stats(cost_ratio),
            "failing_only_cost_ratio": _stats(cost_ratio, failing),
            "exact_match_only_cost_ratio": _stats(cost_ratio, exact_match),
            "fractions": fractions,
            "histogram": histogram,
        },
    }


# ---------------------------------------------------------------------------
# Inference + reporting
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(model, loader, node_info, node_order, num_edges, device):
    model.eval()
    proj_all, gt_all = [], []
    for imgs, _, edge_labels in loader:
        imgs = imgs.to(device)
        _, _, projected, _ = model(imgs, apply_constraints=True)
        proj = node_preds_to_edge_preds(projected.cpu(), node_info, node_order, num_edges)
        proj_all.append((proj > 0.5).numpy().astype(np.uint8))
        gt_all.append(edge_labels.numpy().astype(np.uint8))
    return np.concatenate(proj_all, axis=0), np.concatenate(gt_all, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
        default="../logs/warcraft_ablation_30x30_pw30_ce_only_diagonal/crl_in_training/best_run0.pt")
    parser.add_argument("--data_dir", type=str,
        default="warcraft_shortest_path_oneskin/30x30/preprocessed_4conn")
    parser.add_argument("--grid-size", type=int, default=30)
    parser.add_argument("--processing-order", type=str, default="diagonal")
    parser.add_argument("--out_json", type=str,
        default="../logs/warcraft_ablation_30x30_pw30_ce_only_diagonal/cost_ratio_analysis.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = args.grid_size

    node_info, var_ranges, node_order, total_dim = build_node_categories(N)
    data_dir = Path(args.data_dir)

    test_ds = WarcraftDataset(data_dir, "test", node_info, node_order)
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)
    edges_lookup = np.load(data_dir / "edges.npy")
    vertex_weights = np.load(data_dir / "test_vertex_weights.npy")  # (n, N, N) float16
    print(f"Test samples: {len(test_ds)}  vertex_weights: {vertex_weights.shape} {vertex_weights.dtype}")

    model = ResNet18NodeCategorical(var_ranges=var_ranges, grid_size=N,
                                    processing_order=args.processing_order).to(device)
    model.load_state_dict(torch.load(args.ckpt, weights_only=True, map_location=device))

    pred_edges, gt_edges = collect_predictions(
        model, loader, node_info, node_order, edges_lookup.shape[0], device
    )
    print(f"pred shape={pred_edges.shape}  gt shape={gt_edges.shape}")

    # ----- Sanity check: GT-as-prediction must give cost_ratio == 1.0 -----
    sanity = compute_cost_metrics(gt_edges, gt_edges, vertex_weights, edges_lookup, N)
    sr = np.array(sanity["per_sample"]["cost_ratio"])
    sanity_ok = bool(np.allclose(sr, 1.0, atol=1e-9, equal_nan=False))
    print(f"\n[sanity] GT-vs-GT  ratio  mean={sr.mean():.10f}  "
          f"min={sr.min():.10f}  max={sr.max():.10f}  all==1.0? {sanity_ok}")
    if not sanity_ok:
        deviating = np.where(np.abs(sr - 1.0) > 1e-9)[0]
        print(f"[sanity] FAILED — {len(deviating)} samples with ratio != 1.0; aborting.")
        raise SystemExit(1)

    # ----- Real metric on predictions -----
    out = compute_cost_metrics(pred_edges, gt_edges, vertex_weights, edges_lookup, N)
    agg = out["aggregate"]

    print("\n" + "=" * 70)
    print(f"COST-RATIO METRICS  (ckpt: {Path(args.ckpt).name})")
    print("=" * 70)
    print(f"Total samples : {agg['n_total']}")
    print(f"Failing       : {agg['n_failing']} "
          f"({agg['n_failing']/agg['n_total']*100:.2f}%)")
    print(f"Exact-match   : {agg['n_total']-agg['n_failing']} "
          f"({(agg['n_total']-agg['n_failing'])/agg['n_total']*100:.2f}%)")
    print(f"Invalid (not a path) : {agg['n_invalid']} "
          f"({agg['n_invalid']/agg['n_total']*100:.2f}%)")

    print("\n--- cost_ratio summary ---")
    for k in ["all_samples_cost_ratio", "failing_only_cost_ratio", "exact_match_only_cost_ratio"]:
        s = agg[k]
        print(f"  {k:32s}  n={s['n']:>4d}  mean={s['mean']:.4f}  median={s['median']:.4f}")

    print("\n--- Threshold fractions ---")
    for k, v in agg["fractions"].items():
        print(f"  {k:18s}  {v*100:6.2f}%   ({int(v*agg['n_total'])} / {agg['n_total']})")

    print("\n--- Cost-ratio histogram ---")
    for k, v in agg["histogram"].items():
        print(f"  {k:22s}  {v:>5d}  ({v/agg['n_total']*100:5.2f}%)")

    # Save JSON.
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out["sanity_check"] = {
        "gt_vs_gt_max_abs_dev_from_1.0": float(np.abs(sr - 1.0).max()),
        "all_within_tol_1e-9": sanity_ok,
    }
    out["checkpoint"] = args.ckpt
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
