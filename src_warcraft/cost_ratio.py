import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import WarcraftDataset
from model import ResNet18NodeCategorical
from node_categories import build_node_categories, node_preds_to_edge_preds
from compute_cost_ratio import compute_cost_metrics


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
    parser.add_argument("--ckpt_dir", type=str,
        default="../logs/warcraft_ablation_edge_aux_v2/crl_in_training")
    parser.add_argument("--data_dir", type=str,
        default="warcraft_shortest_path_oneskin/12x12/preprocessed_4conn")
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--processing-order", type=str, default="row",
                        choices=["row", "diagonal"])
    parser.add_argument("--out_json", type=str,
        default="../logs/warcraft_ablation_edge_aux_v2/cost_ratio_multi_seed.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = args.grid_size

    node_info, var_ranges, node_order, total_dim = build_node_categories(N)
    data_dir = Path(args.data_dir)

    test_ds = WarcraftDataset(data_dir, "test", node_info, node_order)
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)
    edges_lookup = np.load(data_dir / "edges.npy")
    vertex_weights = np.load(data_dir / "test_vertex_weights.npy")
    print(f"Grid: {N}x{N}  test samples: {len(test_ds)}  vw: {vertex_weights.shape}")

    ckpt_dir = Path(args.ckpt_dir)
    ckpts = sorted(ckpt_dir.glob("best_run*.pt"))
    if not ckpts:
        raise SystemExit(f"No best_run*.pt found in {ckpt_dir}")
    print(f"Found {len(ckpts)} checkpoints in {ckpt_dir}")

    # Sanity check (one-shot, GT vs GT) — uses only data, no model.
    n_samples = vertex_weights.shape[0]
    dummy_gt = None
    # Actually, gt_edges is what comes out of the loader, easier to do once below.

    per_seed = {}
    all_pred = []
    all_gt = None  # GT is identical across seeds; collect once.

    for ckpt in ckpts:
        seed_tag = ckpt.stem  # e.g. best_run0
        print(f"\n--- {seed_tag} ---")

        model = ResNet18NodeCategorical(
            var_ranges=var_ranges, grid_size=N,
            processing_order=args.processing_order,
        ).to(device)
        model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))

        pred_edges, gt_edges = collect_predictions(
            model, loader, node_info, node_order, edges_lookup.shape[0], device
        )
        if all_gt is None:
            all_gt = gt_edges
            # GT-vs-GT sanity check (only need to do this once).
            sanity = compute_cost_metrics(gt_edges, gt_edges, vertex_weights,
                                          edges_lookup, N)
            sr = np.array(sanity["per_sample"]["cost_ratio"])
            sanity_ok = bool(np.allclose(sr, 1.0, atol=1e-9))
            print(f"  [sanity] GT-vs-GT  max|r-1|={np.abs(sr-1).max():.2e}  ok={sanity_ok}")
            if not sanity_ok:
                raise SystemExit("Sanity check failed.")
        else:
            assert (all_gt == gt_edges).all(), "GT mismatch across seeds!"

        out = compute_cost_metrics(pred_edges, gt_edges, vertex_weights,
                                   edges_lookup, N)
        agg = out["aggregate"]
        ex_match = sum(out["per_sample"]["exact_match"])
        print(f"  exact-match: {ex_match}/{len(test_ds)} = "
              f"{ex_match/len(test_ds)*100:.2f}%")
        print(f"  cost_ratio  mean={agg['all_samples_cost_ratio']['mean']:.4f}  "
              f"median={agg['all_samples_cost_ratio']['median']:.4f}")
        print(f"  ratio==1.0  {agg['fractions']['ratio_eq_1.0']*100:5.2f}%   "
              f"≤1.01 {agg['fractions']['ratio_le_1.01']*100:5.2f}%   "
              f"≤1.05 {agg['fractions']['ratio_le_1.05']*100:5.2f}%   "
              f"≤1.10 {agg['fractions']['ratio_le_1.10']*100:5.2f}%")

        per_seed[seed_tag] = {
            "exact_match_pct": ex_match / len(test_ds) * 100,
            "n_samples": len(test_ds),
            "n_failing": agg["n_failing"],
            "all_samples_cost_ratio": agg["all_samples_cost_ratio"],
            "failing_only_cost_ratio": agg["failing_only_cost_ratio"],
            "fractions": agg["fractions"],
            "histogram": agg["histogram"],
            "per_sample_cost_ratio": out["per_sample"]["cost_ratio"],
        }
        all_pred.append(pred_edges)

    # Aggregate across seeds: per-metric mean ± std over seeds.
    seed_keys = list(per_seed.keys())
    def _arr(getter):
        return np.array([getter(per_seed[k]) for k in seed_keys])

    agg_summary = {
        "exact_match_pct":      {"mean": float(_arr(lambda s: s["exact_match_pct"]).mean()),
                                 "std":  float(_arr(lambda s: s["exact_match_pct"]).std(ddof=0))},
        "ratio_mean":           {"mean": float(_arr(lambda s: s["all_samples_cost_ratio"]["mean"]).mean()),
                                 "std":  float(_arr(lambda s: s["all_samples_cost_ratio"]["mean"]).std(ddof=0))},
        "ratio_median":         {"mean": float(_arr(lambda s: s["all_samples_cost_ratio"]["median"]).mean()),
                                 "std":  float(_arr(lambda s: s["all_samples_cost_ratio"]["median"]).std(ddof=0))},
        "fail_ratio_mean":      {"mean": float(_arr(lambda s: s["failing_only_cost_ratio"]["mean"]).mean()),
                                 "std":  float(_arr(lambda s: s["failing_only_cost_ratio"]["mean"]).std(ddof=0))},
        "fail_ratio_median":    {"mean": float(_arr(lambda s: s["failing_only_cost_ratio"]["median"]).mean()),
                                 "std":  float(_arr(lambda s: s["failing_only_cost_ratio"]["median"]).std(ddof=0))},
        "frac_eq_1.0":          {"mean": float(_arr(lambda s: s["fractions"]["ratio_eq_1.0"]).mean()),
                                 "std":  float(_arr(lambda s: s["fractions"]["ratio_eq_1.0"]).std(ddof=0))},
        "frac_le_1.01":         {"mean": float(_arr(lambda s: s["fractions"]["ratio_le_1.01"]).mean()),
                                 "std":  float(_arr(lambda s: s["fractions"]["ratio_le_1.01"]).std(ddof=0))},
        "frac_le_1.05":         {"mean": float(_arr(lambda s: s["fractions"]["ratio_le_1.05"]).mean()),
                                 "std":  float(_arr(lambda s: s["fractions"]["ratio_le_1.05"]).std(ddof=0))},
        "frac_le_1.10":         {"mean": float(_arr(lambda s: s["fractions"]["ratio_le_1.10"]).mean()),
                                 "std":  float(_arr(lambda s: s["fractions"]["ratio_le_1.10"]).std(ddof=0))},
    }

    # Print summary table.
    print("\n" + "=" * 78)
    print(f"COST-RATIO ACROSS {len(seed_keys)} SEEDS — {ckpt_dir}")
    print("=" * 78)
    print(f"{'metric':<22s} | " + " | ".join(f"{k:>11s}" for k in seed_keys) +
          " || mean ± std")
    print("-" * 78)
    rows = [
        ("exact_match (%)",     "exact_match_pct"),
        ("ratio mean",          "ratio_mean"),
        ("ratio median",        "ratio_median"),
        ("fail-only mean",      "fail_ratio_mean"),
        ("fail-only median",    "fail_ratio_median"),
        ("ratio == 1.0 (%)",    "frac_eq_1.0"),
        ("ratio <= 1.01 (%)",   "frac_le_1.01"),
        ("ratio <= 1.05 (%)",   "frac_le_1.05"),
        ("ratio <= 1.10 (%)",   "frac_le_1.10"),
    ]
    accessors = {
        "exact_match_pct":   lambda s: s["exact_match_pct"],
        "ratio_mean":        lambda s: s["all_samples_cost_ratio"]["mean"],
        "ratio_median":      lambda s: s["all_samples_cost_ratio"]["median"],
        "fail_ratio_mean":   lambda s: s["failing_only_cost_ratio"]["mean"],
        "fail_ratio_median": lambda s: s["failing_only_cost_ratio"]["median"],
        "frac_eq_1.0":       lambda s: s["fractions"]["ratio_eq_1.0"],
        "frac_le_1.01":      lambda s: s["fractions"]["ratio_le_1.01"],
        "frac_le_1.05":      lambda s: s["fractions"]["ratio_le_1.05"],
        "frac_le_1.10":      lambda s: s["fractions"]["ratio_le_1.10"],
    }
    for label, key in rows:
        is_pct = key.startswith("frac") or key == "exact_match_pct"
        scale = 100.0 if key.startswith("frac") else 1.0
        vals = [accessors[key](per_seed[k]) * scale for k in seed_keys]
        m = agg_summary[key]["mean"] * scale
        s = agg_summary[key]["std"] * scale
        cells = " | ".join(f"{v:11.4f}" for v in vals)
        print(f"{label:<22s} | {cells} || {m:7.4f} ± {s:.4f}")

    print("\n--- Cost-ratio histogram (mean count across seeds) ---")
    bin_keys = list(per_seed[seed_keys[0]]["histogram"].keys())
    for bk in bin_keys:
        vals = [per_seed[k]["histogram"][bk] for k in seed_keys]
        m = float(np.mean(vals)); s = float(np.std(vals, ddof=0))
        print(f"  {bk:22s}  mean={m:6.2f}  std={s:5.2f}   per-seed: {vals}")

    # Save full payload.
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ckpt_dir": str(ckpt_dir),
        "data_dir": str(data_dir),
        "grid_size": N,
        "processing_order": args.processing_order,
        "n_seeds": len(seed_keys),
        "seed_tags": seed_keys,
        "per_seed": per_seed,
        "across_seeds_summary": agg_summary,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
