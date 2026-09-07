import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for sub in ("tasks/warcraft", "tasks/warcraft", "examples"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from compute_cost_ratio import _is_valid_src_sink_path, compute_cost_metrics  # noqa: E402

from eval_steps_bridge import edges_lookup_for, load_backbone  # noqa: E402
from eval_warcraft_modes import gt_is_monotone  # noqa: E402
from steps_bridge import TimeAwareBridge, softmax_per_node  # noqa: E402
from steps_walk import StepsWalkDecoder, edge_endpoints_array  # noqa: E402


def main() -> None:
    """Tripwire smoke: N boards through the full steps pipeline, config D, certify on."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--grid-size", type=int, default=12)
    ap.add_argument("--boards", type=int, default=50)
    ap.add_argument("--data-dir", type=Path, default=Path(
        "/home/lk1325/mnist_experiment/warcraft_crl/warcraft_shortest_path_oneskin/"
        "12x12/preprocessed_4conn"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    N = args.grid_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bridge = TimeAwareBridge(N, alphabet="full")
    dec = StepsWalkDecoder(N)
    endpoints = edge_endpoints_array(N)
    src_flat, sink_flat = 0, (N - 1) * N + (N - 1)

    gt = np.load(args.data_dir / "test_edge_labels.npy").astype(np.uint8)[:args.boards]
    maps = np.load(args.data_dir / "test_maps.npy")[:args.boards]
    vw = np.load(args.data_dir / "test_vertex_weights.npy").astype(np.float64)[:args.boards]

    net = load_backbone(args.ckpt, bridge.total_dim, N, device)
    with torch.no_grad():
        imgs = torch.from_numpy(maps).float().div_(255.0).permute(0, 3, 1, 2).to(device)
        probs = softmax_per_node(net(imgs), bridge.var_ranges)
    cell_scores = bridge.cell_scores(probs)

    L = dec.enc.horizon
    pred = np.zeros((len(gt), endpoints.shape[0]), dtype=np.uint8)
    snipped, arrivals, ms = [], [], []
    valid = dead = 0
    for b in range(len(gt)):
        ticks = bridge.tick_scores(cell_scores[b], L)
        t0 = time.perf_counter()
        seq, stats = dec.walk(ticks, selection="argmax", early_cut=True,
                              tie_break="no-backtrack", forbid_reversal=True)
        erased, snip, vec = dec.decode(seq)
        ms.append((time.perf_counter() - t0) * 1e3)
        pred[b] = vec
        snipped.append(snip)
        arrivals.append(stats.arrival)
        dead += stats.dead_ends
        if _is_valid_src_sink_path(vec.astype(bool), endpoints, src_flat, sink_flat):
            valid += 1

    full = compute_cost_metrics(pred, gt, vw, edges_lookup_for(N), N)
    cost = full["aggregate"]
    ratio = np.asarray(full["per_sample"]["cost_ratio"], dtype=np.float64)
    em = (pred.astype(bool) == gt.astype(bool)).all(axis=1)
    ham = (pred.astype(bool) != gt.astype(bool)).sum(axis=1)
    mono = gt_is_monotone(gt, N)

    def slice_stats(mask):
        """EM / optimal / ratio restricted to a census slice."""
        if not mask.any():
            return {"n": 0}
        r = ratio[mask]
        finite = r[np.isfinite(r)]
        return {
            "n": int(mask.sum()),
            "EM_pct": 100.0 * float(em[mask].mean()),
            "optimal_pct": 100.0 * float((np.abs(r - 1.0) < 1e-6).mean()),
            "consistent_pct": 100.0 * float(np.isfinite(r).mean()),
            "ratio_mean": float(finite.mean()) if finite.size else float("nan"),
        }

    out = {
        "ckpt": str(args.ckpt), "boards": len(gt), "config": "D (time-aware + no-backtrack)",
        "certify": True,
        "certificates_passed": valid,
        "certificate_failures": len(gt) - valid,
        "dead_ends": dead,
        "erasure_rate_pct": 100.0 * float(np.mean(np.asarray(snipped) > 0)),
        "mean_cells_snipped": float(np.mean(snipped)),
        "mean_arrival": float(np.mean(arrivals)),
        "ms_per_board": float(np.mean(ms)),
        "EM_pct": 100.0 * float(em.mean()),
        "exact_fraction": float(em.mean()),
        "hamming_mean": float(ham.mean()),
        "optimal_pct": 100.0 * cost["fractions"]["ratio_eq_1.0"],
        "within_1pct": 100.0 * cost["fractions"]["ratio_le_1.01"],
        "within_5pct": 100.0 * cost["fractions"]["ratio_le_1.05"],
        "within_10pct": 100.0 * cost["fractions"]["ratio_le_1.10"],
        "cost_ratio_mean": cost["all_samples_cost_ratio"]["mean"],
        "cost_ratio_median": cost["all_samples_cost_ratio"]["median"],
        "histogram": cost["histogram"],
        "census": {"monotone_gt": slice_stats(mono),
                   "non_monotone_gt": slice_stats(~mono)},
    }
    print(json.dumps(out, indent=2), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
