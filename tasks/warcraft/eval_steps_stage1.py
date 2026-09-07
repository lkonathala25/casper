import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tasks/warcraft", "tasks/warcraft"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from compute_cost_ratio import _is_valid_src_sink_path, compute_cost_metrics  # noqa: E402

from eval_steps_bridge import edges_lookup_for, load_backbone  # noqa: E402
from steps_bridge import (  # noqa: E402
    PlainBroadcastBridge,
    TimeAwareBridge,
    softmax_per_node,
)
from steps_walk import StepsWalkDecoder, edge_endpoints_array  # noqa: E402

BRIDGE_NOTE = "bridge readings evaluated because the reference's λ=0 definition is unavailable"
INFERENCE_NOTE = "inference-only by our choice; the tex specifies no training regime."
REFERENCE = {"config": "lambda=0", "n": "6/6", "optimal_pct": 68.3,
             "optimal_std": 2.9, "exact": 0.574, "exact_std": 0.042}

# The adopted tie-break is active in every cell; B and D additionally apply the
# stronger, unadopted reversal ban, so the screen separates decoder-side forward
# pressure (B, D) from bridge-side forward pressure (C, D).
CELLS = [
    ("A", "plain broadcast", False),
    ("B", "plain broadcast", True),
    ("C", "time-aware", False),
    ("D", "time-aware", True),
]


def run_cell(dec, bridge, cell_scores, gt, vw, lut, endpoints, N, forbid_reversal):
    """Walk -> early cut -> loop erasure -> certificate for every board."""
    L = dec.enc.horizon
    src_flat, sink_flat = 0, (N - 1) * N + (N - 1)
    n = cell_scores.shape[0]
    pred = np.zeros((n, endpoints.shape[0]), dtype=np.uint8)
    snipped, arrivals, ms = [], [], []
    valid = decisions = strict_rev = ties = dead = 0

    for b in range(n):
        ticks = bridge.tick_scores(cell_scores[b], L)
        t0 = time.perf_counter()
        seq, stats = dec.walk(ticks, selection="argmax", early_cut=True,
                              tie_break="no-backtrack",
                              forbid_reversal=forbid_reversal)
        erased, snip, vec = dec.decode(seq)
        ms.append((time.perf_counter() - t0) * 1e3)
        pred[b] = vec
        snipped.append(snip)
        arrivals.append(stats.arrival)
        decisions += stats.decisions
        strict_rev += stats.strict_reversals
        ties += stats.ties_at_top
        dead += stats.dead_ends
        if _is_valid_src_sink_path(vec.astype(bool), endpoints, src_flat, sink_flat):
            valid += 1

    cost = compute_cost_metrics(pred, gt, vw, lut, N)["aggregate"]
    em = (pred.astype(bool) == gt.astype(bool)).all(axis=1)
    hamming = (pred.astype(bool) != gt.astype(bool)).sum(axis=1)
    return {
        "boards": n,
        "EM_pct": 100.0 * float(em.mean()),
        "exact_fraction": float(em.mean()),
        "optimal_pct": 100.0 * cost["fractions"]["ratio_eq_1.0"],
        "hamming_mean": float(hamming.mean()),
        "consistent_pct": 100.0 * valid / n,
        "certificate_failures": n - valid,
        "cost_ratio_mean": cost["all_samples_cost_ratio"]["mean"],
        "cost_ratio_n_finite": cost["all_samples_cost_ratio"]["n"],
        "ms_per_board": float(np.mean(ms)),
        "erasure_rate_pct": 100.0 * float(np.mean(np.asarray(snipped) > 0)),
        "mean_cells_snipped": float(np.mean(snipped)),
        "mean_arrival": float(np.mean(arrivals)),
        "decision_steps": decisions,
        "ties_at_top": ties,
        "strict_reversals": strict_rev,
        "strict_reversal_rate_pct": 100.0 * strict_rev / max(decisions, 1),
        "dead_ends": dead,
    }


def main() -> None:
    """Stage 1: the 2x2 bridge-reading x decoder screen on one seed."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-size", type=int, default=12)
    ap.add_argument("--data-dir", type=Path, default=Path(
        "/home/lk1325/mnist_experiment/warcraft_crl/warcraft_shortest_path_oneskin/"
        "12x12/preprocessed_4conn"))
    ap.add_argument("--ckpt", type=Path, default=Path(
        "/home/lk1325/mnist_experiment/logs/warcraft_ablation_edge_aux_v2/"
        "crl_in_training/best_run0.pt"))
    ap.add_argument("--boards", type=int, default=0, help="0 = whole test set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "logs" / "steps_phase5_stage1.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    N = args.grid_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{BRIDGE_NOTE}\n{INFERENCE_NOTE}\ndevice={device}", flush=True)

    gt = np.load(args.data_dir / "test_edge_labels.npy").astype(np.uint8)
    maps = np.load(args.data_dir / "test_maps.npy")
    vw = np.load(args.data_dir / "test_vertex_weights.npy").astype(np.float64)
    if args.boards:
        gt, maps, vw = gt[:args.boards], maps[:args.boards], vw[:args.boards]

    lut = edges_lookup_for(N)
    endpoints = edge_endpoints_array(N)
    dec = StepsWalkDecoder(N)
    bridges = {
        "plain broadcast": PlainBroadcastBridge(N, alphabet="full"),
        "time-aware": TimeAwareBridge(N, alphabet="full"),
    }

    net = load_backbone(args.ckpt, bridges["plain broadcast"].total_dim, N, device)
    scores_all = []
    with torch.no_grad():
        for lo in range(0, len(maps), 250):
            imgs = torch.from_numpy(maps[lo:lo + 250]).float().div_(255.0)
            imgs = imgs.permute(0, 3, 1, 2).to(device)
            probs = softmax_per_node(net(imgs), bridges["plain broadcast"].var_ranges)
            scores_all.append(bridges["plain broadcast"].cell_scores(probs))
    cell_scores = np.concatenate(scores_all, axis=0)
    print(f"boards={len(gt)}  ckpt={args.ckpt.name}", flush=True)

    result = {
        "stage": 1,
        "bridge_note": BRIDGE_NOTE,
        "inference_note": INFERENCE_NOTE,
        "reference": REFERENCE,
        "caveat": "Stage 1 is single-seed and can indicate neighbourhood only, "
                  "not a match.",
        "grid": N, "seed": args.seed, "ckpt": str(args.ckpt),
        "penalty_strength": TimeAwareBridge.PENALTY_STRENGTH,
        "cells": {},
    }
    for name, bridge_name, forbid_reversal in CELLS:
        row = run_cell(dec, bridges[bridge_name], cell_scores, gt, vw, lut,
                       endpoints, N, forbid_reversal)
        row["bridge"] = bridge_name
        row["decoder"] = ("no-backtrack (reversal ban)" if forbid_reversal
                          else "adopted tie-break only")
        result["cells"][name] = row
        print(f"{name}: EM={row['EM_pct']:.1f}%  optimal={row['optimal_pct']:.1f}%  "
              f"strict_rev={row['strict_reversal_rate_pct']:.1f}%", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
