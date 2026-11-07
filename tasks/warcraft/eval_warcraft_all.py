import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_ROOT / "casper")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from compute_cost_ratio import compute_cost_metrics  # noqa: E402
from data import WarcraftDataset  # noqa: E402
from model import ResNet18NodeCategorical  # noqa: E402
from node_categories import build_node_categories, node_preds_to_edge_preds  # noqa: E402

# Each reported row: the run folder it reads (its own name unless "dir" says
# otherwise), the checkpoint inside each seed folder (first existing candidate),
# how a prediction is decoded, and the metrics file it writes. CASPER runs keep
# their checkpoint under the --casper-mode they trained with. Train-Only is read
# off the full-CASPER run's raw-validation checkpoint with the argmax decode,
# into its own metrics file. "resnet" is the unconstrained ResNet-18.
_RESNET_CKPT = ("none/best_run0.pt", "no_crl/best_run0.pt")
METHODS = {
    "casper_train_infer_l2_penalty1": {"ckpt": ("train_infer/best_run0.pt",),
                                       "decode": "steps_projection"},
    "casper_train_infer_l2_penalty0": {"ckpt": ("train_infer/best_run0.pt",),
                                       "decode": "steps_projection"},
    "casper_train_only": {"dir": "casper_train_infer_l2_penalty1",
                          "ckpt": ("train_infer/best_raw_run0.pt",), "decode": "argmax",
                          "metrics": "metrics_train_only.json"},
    "casper_infer_only": {"ckpt": ("infer_only/best_run0.pt",), "decode": "steps_projection"},
    "resnet": {"ckpt": _RESNET_CKPT, "decode": "argmax"},
}

# A trained ResNet-18 reaches the eval under three different key layouts.
STATE_PREFIXES = (
    "backbone.",
    "",
)

ALPHA_STOP = 1.5
ALPHA_STEP = 0.01
EQ_TOL = 1e-6


def find_checkpoint(run_dir, method):
    """Path to a run's reported checkpoint, or None when the run has not produced one.

    Rows read off another method's folder ("dir") take only their named
    checkpoints, never a guessed fallback.
    """
    run_dir = Path(run_dir)
    named = next((run_dir / c for c in METHODS[method]["ckpt"] if (run_dir / c).exists()), None)
    if named is not None or "dir" in METHODS[method]:
        return named
    fallbacks = sorted(run_dir.rglob("best_run*.pt"))
    return next((p for p in fallbacks if p.exists()), None)


def load_backbone_weights(ckpt_path):
    """ResNet-18 weights from a CASPER or ResNet checkpoint, keyed as torchvision expects.

    Both save the full model under 'backbone.'; the prefix is stripped.
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = state.get("model", state) if isinstance(state, dict) and "model" in state else state
    for prefix in STATE_PREFIXES:
        stripped = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if "conv1.weight" in stripped and "fc.weight" in stripped:
            return stripped, prefix or "(none)"
    raise ValueError(
        f"{ckpt_path} holds no ResNet-18 weights under any known prefix; "
        f"keys start with {sorted({k.split('.')[0] for k in state})[:5]}")


def build_model(method, var_ranges, grid_size, steps_artifact, device):
    """The method's network, with its constraint layer when that is how it predicts."""
    constraint_mode = "steps" if METHODS[method]["decode"] == "steps_projection" else "none"
    model = ResNet18NodeCategorical(
        var_ranges=var_ranges, grid_size=grid_size, constraint_mode=constraint_mode,
        alphabet="full", steps_artifact=steps_artifact if constraint_mode == "steps" else None,
    ).to(device)
    if constraint_mode == "steps" and not model.constraint_layer.from_artifact:
        raise RuntimeError(f"steps layer was not built from the compiled artifact {steps_artifact}; "
                           f"evaluation must project with the same layer training uses")
    return model.eval()


def predict_edges(method, model, loader, node_info, node_order, num_edges, device):
    """(n_samples, n_edges) uint8 predictions, decoded the way this method reports them."""
    decode = METHODS[method]["decode"]
    preds = []
    with torch.no_grad():
        for imgs, _node_labels, _edge_labels in loader:
            imgs = imgs.to(device)
            _logits, probs, projected, _ = model(
                imgs, apply_constraints=(decode == "steps_projection"))
            out = projected if decode == "steps_projection" else probs
            preds.append(node_preds_to_edge_preds(
                out.cpu(), node_info, node_order, num_edges).numpy())
    return np.concatenate(preds).astype(np.uint8)


def alpha_curve(cost_ratio, stop=ALPHA_STOP, step=ALPHA_STEP):
    """Cumulative cost-optimality curve: fraction of predictions within a factor alpha.

    Alphas run from 1.0 to `stop`. Invalid predictions carry an infinite ratio
    (compute_cost_metrics), so they never satisfy a threshold but stay in the
    denominator, as for the headline Optimal numbers.
    """
    alphas = np.round(np.arange(1.0, stop + step / 2, step), 4)
    ratio = np.asarray(cost_ratio, dtype=np.float64)
    return {"alpha": alphas.tolist(),
            "fraction": [float((ratio <= a + EQ_TOL).mean()) for a in alphas]}


def paper_metrics(pred_edges, gt_edges, vertex_weights, edges_lookup, grid_size,
                  optimal_costs=None):
    """Table 2, Tables 7-8 and the Figure 3 curve for one run's predictions.

    Optimal counts predictions whose path cost equals the ground-truth minimum;
    the tolerance columns and the curve count those within a factor of it.
    Predictions that are not valid paths count as neither, but stay in the
    denominator everywhere.
    """
    out = compute_cost_metrics(pred_edges, gt_edges, vertex_weights, edges_lookup, grid_size)
    per = out["per_sample"]
    ratio = np.asarray(per["cost_ratio"], dtype=np.float64)
    valid = np.asarray(per["is_valid_path"], dtype=bool)
    pred = np.asarray(pred_edges).astype(bool)
    gt = np.asarray(gt_edges).astype(bool)
    n = pred.shape[0]

    metrics = {
        "n_test": int(n),
        "optimal": 100.0 * float((np.abs(ratio - 1.0) < EQ_TOL).mean()),
        "optimal_le_1pct": 100.0 * float((ratio <= 1.01 + EQ_TOL).mean()),
        "optimal_le_5pct": 100.0 * float((ratio <= 1.05 + EQ_TOL).mean()),
        "optimal_le_10pct": 100.0 * float((ratio <= 1.10 + EQ_TOL).mean()),
        "consistent": 100.0 * float(valid.mean()),
        "exact_match": 100.0 * float((pred == gt).all(axis=1).mean()),
        "hamming": 100.0 * float((pred == gt).mean()),
        "mean_cost_ratio_valid": float(np.mean(ratio[np.isfinite(ratio)])) if valid.any() else float("nan"),
        "cumulative_cost_optimality": alpha_curve(ratio),
    }
    if optimal_costs is not None:
        gt_cost = np.asarray(per["gt_cost"], dtype=np.float64)
        stored = np.asarray(optimal_costs, dtype=np.float64)
        metrics["gt_cost_max_abs_diff_vs_stored"] = float(np.max(np.abs(gt_cost - stored)))
    metrics["per_sample"] = {
        "cost_ratio": [None if not np.isfinite(r) else float(r) for r in ratio],
        "pred_cost": [float(c) for c in per["pred_cost"]],
        "gt_cost": [float(c) for c in per["gt_cost"]],
        "is_valid_path": valid.tolist(),
    }
    return metrics


def evaluate_run(method, run_dir, data_dir, grid_size, device, batch_size=128,
                 num_workers=4):
    """Score one trained run on the test split and write the method's metrics file in its folder."""
    ckpt = find_checkpoint(run_dir, method)
    if ckpt is None:
        return None
    node_info, var_ranges, node_order, _total_dim = build_node_categories(
        grid_size, alphabet="full")
    data_dir = Path(data_dir)
    edges = np.load(data_dir / "edges.npy")
    steps_artifact = (_ROOT / "constraints" / "warcraft" / "steps_artifacts"
                      / f"steps_{grid_size}_trimmed_compiled.json")

    model = build_model(method, var_ranges, grid_size, steps_artifact, device)
    state, prefix = load_backbone_weights(ckpt)
    model.backbone.load_state_dict(state)

    test_ds = WarcraftDataset(data_dir, "test", node_info, node_order)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    pred = predict_edges(method, model, loader, node_info, node_order,
                         int(edges.shape[0]), device)

    metrics = paper_metrics(
        pred, test_ds.edge_labels, np.load(data_dir / "test_vertex_weights.npy"),
        edges, grid_size, optimal_costs=np.load(data_dir / "test_costs.npy"))
    metrics["run"] = {"method": method, "decode": METHODS[method]["decode"],
                      "checkpoint": str(ckpt), "state_prefix": prefix,
                      "run_dir": str(run_dir), "grid_size": grid_size,
                      "data_dir": str(data_dir)}
    (Path(run_dir) / METHODS[method].get("metrics", "metrics.json")).write_text(
        json.dumps(metrics, indent=2))
    return metrics


def summarise(results):
    """Mean and sd over seeds per method, with the seed-averaged curve."""
    keys = ["optimal", "optimal_le_1pct", "optimal_le_5pct", "optimal_le_10pct",
            "consistent", "exact_match", "hamming", "mean_cost_ratio_valid"]
    summary = {}
    for method, runs in results.items():
        if not runs:
            continue
        stats = {}
        for key in keys:
            vals = np.array([r[key] for r in runs.values()], dtype=np.float64)
            stats[key] = {"mean": float(vals.mean()),
                          "sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                          "per_seed": {s: float(r[key]) for s, r in runs.items()}}
        curves = np.array([r["cumulative_cost_optimality"]["fraction"] for r in runs.values()])
        stats["cumulative_cost_optimality"] = {
            "alpha": next(iter(runs.values()))["cumulative_cost_optimality"]["alpha"],
            "fraction_mean": curves.mean(axis=0).tolist(),
            "fraction_sd": (curves.std(axis=0, ddof=1) if len(curves) > 1
                            else np.zeros(curves.shape[1])).tolist(),
        }
        stats["seeds"] = sorted(runs)
        summary[method] = stats
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=str, default=str(_ROOT / "runs" / "warcraft"))
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Defaults to data/warcraft/<grid>x<grid>/preprocessed_4conn.")
    parser.add_argument("--methods", type=str, nargs="*", default=sorted(METHODS),
                        choices=sorted(METHODS))
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Defaults to every seed<N> folder present.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out", type=str, default=None,
                        help="Summary JSON; defaults to <runs-root>/<grid>x<grid>/summary.json")
    args = parser.parse_args()

    grid = f"{args.grid_size}x{args.grid_size}"
    data_dir = Path(args.data_dir) if args.data_dir else (
        _ROOT / "data" / "warcraft" / grid / "preprocessed_4conn")
    root = Path(args.runs_root) / grid
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for method in args.methods:
        method_dir = root / METHODS[method].get("dir", method)
        seed_dirs = sorted(method_dir.glob("seed*")) if method_dir.is_dir() else []
        results[method] = {}
        for seed_dir in seed_dirs:
            seed = seed_dir.name.replace("seed", "")
            if args.seeds is not None and int(seed) not in args.seeds:
                continue
            metrics = evaluate_run(method, seed_dir, data_dir, args.grid_size, device,
                                   args.batch_size, args.num_workers)
            if metrics is None:
                print(f"{method} seed {seed}: no checkpoint yet, skipped")
                continue
            results[method][seed] = metrics
            print(f"{method:12s} seed {seed}: optimal {metrics['optimal']:5.1f}%  "
                  f"consistent {metrics['consistent']:5.1f}%  "
                  f"EM {metrics['exact_match']:5.1f}%  hamming {metrics['hamming']:5.1f}%  "
                  f"<=1/5/10% {metrics['optimal_le_1pct']:5.1f}/"
                  f"{metrics['optimal_le_5pct']:5.1f}/{metrics['optimal_le_10pct']:5.1f}")

    summary = summarise(results)
    out = Path(args.out) if args.out else root / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary over {sum(len(r) for r in results.values())} runs -> {out}")
    for method, stats in summary.items():
        print(f"  {method:12s} optimal {stats['optimal']['mean']:5.1f} ± {stats['optimal']['sd']:4.1f}  "
              f"consistent {stats['consistent']['mean']:5.1f} ± {stats['consistent']['sd']:4.1f}  "
              f"(seeds {', '.join(stats['seeds'])})")


if __name__ == "__main__":
    main()
