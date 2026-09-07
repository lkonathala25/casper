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

from node_categories import (  # noqa: E402
    UnrepresentablePatternError,
    build_node_categories,
    edge_labels_to_node_labels,
)

from steps_layer import StepsProjectionLayer, one_hot_from_categories  # noqa: E402


def main() -> None:
    """Run the Phase A gates on the reverse bridge and the steps layer."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-size", type=int, default=12)
    ap.add_argument("--data-dir", type=Path, default=Path(
        "/home/lk1325/mnist_experiment/warcraft_crl/warcraft_shortest_path_oneskin/"
        "12x12/preprocessed_4conn"))
    ap.add_argument("--out", type=Path,
                    default=ROOT / "logs" / "steps_layer_phaseA_gates.json")
    args = ap.parse_args()

    N = args.grid_size
    node_info, var_ranges, node_order, total_dim = build_node_categories(
        N, alphabet="full")
    layer = StepsProjectionLayer(N, alphabet="full")
    gt = np.load(args.data_dir / "test_edge_labels.npy").astype(np.uint8)
    gt_cats = edge_labels_to_node_labels(gt, node_info, node_order)
    print(f"boards={len(gt)} total_dim={total_dim}", flush=True)

    result = {"grid": N, "total_dim": total_dim, "boards": len(gt),
              "bridge": layer.bridge_name, "forbid_reversal": layer.forbid_reversal}

    # Gate 1 — round trip on ground truth, all boards.
    onehot = one_hot_from_categories(gt_cats, var_ranges)
    matched = 0
    mismatch_examples = []
    for lo in range(0, len(gt), 100):
        block = onehot[lo:lo + 100]
        proj, info = layer(block)
        got = np.zeros((block.shape[0], len(node_order)), dtype=np.int64)
        off = 0
        for k, vr in enumerate(var_ranges):
            got[:, k] = proj[:, off:off + vr].argmax(dim=1).numpy()
            off += vr
        ok = (got == gt_cats[lo:lo + 100]).all(axis=1)
        matched += int(ok.sum())
        for b in np.nonzero(~ok)[0][:2]:
            if len(mismatch_examples) < 5:
                diff = np.nonzero(got[b] != gt_cats[lo + b])[0]
                mismatch_examples.append(
                    {"board": int(lo + b), "n_nodes_differing": int(diff.size),
                     "first_nodes": diff[:6].tolist()})
    result["gate1_roundtrip"] = {
        "boards": len(gt), "exact_category_match": matched,
        "pct": 100.0 * matched / len(gt), "mismatch_examples": mismatch_examples,
    }
    print(json.dumps(result["gate1_roundtrip"], indent=2), flush=True)

    # Gate 2 — shape and per-node distribution validity.
    probs = torch.softmax(torch.randn(16, total_dim) * 2, dim=1)
    off = 0
    for vr in var_ranges:
        probs[:, off:off + vr] = torch.softmax(probs[:, off:off + vr] * 5, dim=1)
        off += vr
    proj, info = layer(probs)
    sums, mins = [], []
    off = 0
    for vr in var_ranges:
        sums.append(proj[:, off:off + vr].sum(dim=1))
        mins.append(proj[:, off:off + vr].min())
        off += vr
    sums = torch.stack(sums)
    result["gate2_shape_consistency"] = {
        "shape": list(proj.shape),
        "shape_ok": list(proj.shape) == [16, total_dim],
        "nonnegative": bool(min(mins) >= 0.0),
        "min_value": float(min(mins)),
        "per_node_sum_max_abs_dev_from_1": float((sums - 1.0).abs().max()),
        "per_node_sums_ok": bool((sums - 1.0).abs().max() < 1e-4),
    }
    print(json.dumps(result["gate2_shape_consistency"], indent=2), flush=True)

    # Gate 3 — determinism.
    a, _ = layer(probs)
    b, _ = layer(probs)
    result["gate3_determinism"] = {
        "identical": bool(torch.equal(a, b)),
        "max_abs_diff": float((a - b).abs().max()),
    }
    print(json.dumps(result["gate3_determinism"], indent=2), flush=True)

    # Gate 4 — cost per 128-board batch, single pass, no certificate.
    batch = torch.softmax(torch.randn(128, total_dim) * 2, dim=1)
    off = 0
    for vr in var_ranges:
        batch[:, off:off + vr] = torch.softmax(batch[:, off:off + vr] * 5, dim=1)
        off += vr
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        layer(batch)
        times.append((time.perf_counter() - t0) * 1e3)
    result["gate4_cost"] = {
        "batch": 128, "certify": layer.certify,
        "ms_median": float(np.median(times)),
        "ms_min": float(min(times)), "ms_max": float(max(times)),
        "projected_s_per_epoch_79_batches": float(np.median(times) * 79 / 1e3),
    }
    print(json.dumps(result["gate4_cost"], indent=2), flush=True)

    # The reverse bridge is fail-loud on unrepresentable patterns; confirm the
    # guard is live rather than assuming the happy path proves it.
    try:
        edge_labels_to_node_labels(
            np.ones((1, layer.num_edges), dtype=np.int64), node_info, node_order)
        result["reverse_bridge_fail_loud"] = False
    except UnrepresentablePatternError:
        result["reverse_bridge_fail_loud"] = True
    print(f"reverse bridge fail-loud: {result['reverse_bridge_fail_loud']}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
