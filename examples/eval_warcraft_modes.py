from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "src_warcraft"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from edge_bridge import embed_monotone_head_into_full  # noqa: E402
from lazy_exact_layer import build_constraint_layer  # noqa: E402
from model import is_valid_path  # noqa: E402
from node_categories import build_node_categories, node_preds_to_edge_preds  # noqa: E402


def edges_lookup_for(N):
    n_w, n_right = N - 1, N * (N - 1)
    rows = [None] * (2 * N * (N - 1))
    for i in range(N):
        for j in range(N - 1):
            rows[i * n_w + j] = (i, j, i, j + 1)
    for i in range(N - 1):
        for j in range(N):
            rows[n_right + i * N + j] = (i, j, i + 1, j)
    return np.array(rows)


def vertex_cost(edge_mask, vw_flat, endpoints, src, sink):
    visited = np.unique(np.concatenate([
        endpoints[edge_mask].ravel(), np.array([src, sink], dtype=np.int64)]))
    return float(vw_flat[visited].sum())


def gt_is_monotone(gt, N):
    """Per-instance monotonicity of the ground-truth path (the census slice)."""
    lookup = edges_lookup_for(N)
    out = np.zeros(len(gt), dtype=bool)
    for b, row in enumerate(gt):
        adj = {}
        for idx in np.nonzero(row)[0]:
            r1, c1, r2, c2 = lookup[idx]
            adj.setdefault((r1, c1), []).append((r2, c2))
            adj.setdefault((r2, c2), []).append((r1, c1))
        cur, prev, mono = (0, 0), None, True
        for _ in range(2 * N * N):
            nxt = [x for x in adj.get(cur, ()) if x != prev]
            if not nxt:
                break
            if nxt[0][0] < cur[0] or nxt[0][1] < cur[1]:
                mono = False
            prev, cur = cur, nxt[0]
            if cur == (N - 1, N - 1):
                break
        out[b] = mono
    return out


def load_backbone(ckpt, total_dim, N, device):
    sd = torch.load(ckpt, weights_only=True, map_location=device)
    for key in ("nn_state_dict", "model_state", "model_state_dict", "state_dict"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
            break
    sd = {k: v for k, v in sd.items() if k.startswith("backbone.")}
    if sd["backbone.fc.weight"].shape[0] != total_dim:
        sd = embed_monotone_head_into_full(sd, N)
    net = models.resnet18(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, total_dim)
    net.load_state_dict({k[len("backbone."):]: v for k, v in sd.items()})
    return net.to(device).eval()


def softmax_per_node(logits, var_ranges):
    out = torch.zeros_like(logits)
    off = 0
    for vr in var_ranges:
        out[:, off:off + vr] = F.softmax(logits[:, off:off + vr], dim=1)
        off += vr
    return out


def score(pred, gt, vw, lookup, N, mask=None):
    if mask is None:
        mask = np.ones(len(gt), dtype=bool)
    p, g, w = pred[mask], gt[mask], vw[mask]
    n = len(g)
    if n == 0:
        return {}
    endpoints = np.stack([lookup[:, 0] * N + lookup[:, 1],
                          lookup[:, 2] * N + lookup[:, 3]], axis=1)
    src, sink = 0, (N - 1) * N + (N - 1)
    ratios = np.empty(n)
    for i in range(n):
        f = w[i].ravel()
        ratios[i] = (vertex_cost(p[i].astype(bool), f, endpoints, src, sink)
                     / max(vertex_cost(g[i].astype(bool), f, endpoints, src, sink), 1e-12))
    consistent = np.array([is_valid_path(p[i], lookup, N) for i in range(n)])
    return {
        "n": int(n),
        "exact": float((p == g).all(axis=1).mean() * 100),
        "hamming": float((p == g).mean() * 100),
        "consistent": float(consistent.mean() * 100),
        "optimal": float((ratios <= 1 + 1e-6).mean() * 100),
        "ratio_le_1.01": float((ratios <= 1.01 + 1e-6).mean() * 100),
        "ratio_le_1.05": float((ratios <= 1.05 + 1e-6).mean() * 100),
        "ratio_le_1.10": float((ratios <= 1.10 + 1e-6).mean() * 100),
        "ratio_mean": float(ratios.mean()),
        "ratio_median": float(np.median(ratios)),
    }


def run_mode(mode, ckpts, maps, gt, vw, N, device, batch_size, max_rounds):
    alphabet = "monotone" if mode == "monotone" else "full"
    node_info, var_ranges, node_order, total_dim = build_node_categories(
        N, alphabet=alphabet)
    layer = build_constraint_layer(mode, grid_size=N, alphabet=alphabet,
                                   max_rounds=max_rounds)
    lookup = edges_lookup_for(N)
    per_ckpt = {}

    for ck in ckpts:
        net = load_backbone(ck, total_dim, N, device)
        preds = []
        proj_time = 0.0
        rounds_hist, rings, capped, cert_fail = Counter(), 0, 0, 0

        with torch.no_grad():
            for s in range(0, len(maps), batch_size):
                imgs = torch.from_numpy(maps[s:s + batch_size]).float().div_(255.0)
                imgs = imgs.permute(0, 3, 1, 2).to(device)
                probs = softmax_per_node(net(imgs), var_ranges)

                t0 = time.perf_counter()
                projected, info = layer(probs.detach().cpu())
                proj_time += time.perf_counter() - t0

                if mode == "lazy_exact":
                    for lg in info["per_board"]:
                        rounds_hist[lg["rounds"]] += 1
                        rings += sum(lg["rings_per_round"])
                        capped += 1 if lg["capped"] else 0
                        cert_fail += 0 if lg["certified"] else 1
                e = node_preds_to_edge_preds(projected, node_info, node_order,
                                             gt.shape[1])
                preds.append((e > 0.5).numpy().astype(np.uint8))

        pred = np.concatenate(preds, axis=0)
        entry = {
            "all": score(pred, gt, vw, lookup, N),
            "ms_per_board": 1000.0 * proj_time / len(maps),
        }
        if mode == "lazy_exact":
            entry["machinery"] = {
                "rounds_histogram": {str(k): v for k, v in sorted(rounds_hist.items())},
                "boards_needing_refinement": sum(
                    c for r, c in rounds_hist.items() if r > 1),
                "rings_found": rings,
                "capped_boards": capped,
                "certification_failures": cert_fail,
            }
        per_ckpt[ck.name] = (entry, pred)

    return per_ckpt, layer


def agg(values):
    return {"mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, default=Path(
        "/home/lk1325/mnist_experiment/logs/warcraft_ablation_edge_aux_v2/crl_in_training"))
    ap.add_argument("--data-dir", type=Path, default=Path(
        "/home/lk1325/mnist_experiment/warcraft_crl/warcraft_shortest_path_oneskin/"
        "12x12/preprocessed_4conn"))
    ap.add_argument("--grid-size", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=250)
    ap.add_argument("--max-rounds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=Path, default=ROOT / "logs" /
                    "warcraft_eval_modes_12x12.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = args.grid_size

    maps = np.load(args.data_dir / "test_maps.npy")
    gt = np.load(args.data_dir / "test_edge_labels.npy").astype(np.uint8)
    vw = np.load(args.data_dir / "test_vertex_weights.npy").astype(np.float64)
    ckpts = sorted(args.ckpt_dir.glob("best_run*.pt"))
    lookup = edges_lookup_for(N)

    mono_mask = gt_is_monotone(gt, N)
    print(f"grid {N}x{N}  test={len(gt)}  seeds={len(ckpts)}  device={device}")
    print(f"census: monotone GT={mono_mask.sum()}  non-monotone={(~mono_mask).sum()}")

    out = {"census": {"monotone": int(mono_mask.sum()),
                      "non_monotone": int((~mono_mask).sum())},
           "seed": args.seed, "modes": {}}

    for mode in ("monotone", "lazy_exact"):
        per_ckpt, layer = run_mode(mode, ckpts, maps, gt, vw, N, device,
                                   args.batch_size, args.max_rounds)
        slices = {"all": None, "monotone_gt": mono_mask, "non_monotone_gt": ~mono_mask}
        mode_out = {"per_ckpt": {}, "summary": {}}

        for name, entry_pred in per_ckpt.items():
            entry, pred = entry_pred
            row = {"ms_per_board": entry["ms_per_board"]}
            for sl, mask in slices.items():
                row[sl] = score(pred, gt, vw, lookup, N, mask)
            if "machinery" in entry:
                row["machinery"] = entry["machinery"]
            mode_out["per_ckpt"][name] = row

        keys = ["exact", "optimal", "hamming", "consistent", "ratio_mean",
                "ratio_median", "ratio_le_1.01", "ratio_le_1.05", "ratio_le_1.10"]
        for sl in slices:
            mode_out["summary"][sl] = {
                k: agg([mode_out["per_ckpt"][c][sl][k] for c in mode_out["per_ckpt"]])
                for k in keys}
        mode_out["summary"]["ms_per_board"] = agg(
            [mode_out["per_ckpt"][c]["ms_per_board"] for c in mode_out["per_ckpt"]])
        if mode == "lazy_exact":
            mode_out["lifetime"] = {
                "empty_feasible": layer.lifetime_empty_feasible,
                "boards": layer.lifetime_boards,
                "capped": layer.lifetime_capped_boards,
                "rings_forbidden": layer.lifetime_rings_forbidden,
            }
        out["modes"][mode] = mode_out

        print(f"\n===== {mode} =====")
        for sl in slices:
            s = mode_out["summary"][sl]
            n = mode_out["per_ckpt"][list(mode_out["per_ckpt"])[0]][sl]["n"]
            print(f"  [{sl}] n={n}")
            print(f"    EM {s['exact']['mean']:6.2f}±{s['exact']['std']:.2f}   "
                  f"Optimal {s['optimal']['mean']:6.2f}±{s['optimal']['std']:.2f}   "
                  f"Hamming {s['hamming']['mean']:6.2f}±{s['hamming']['std']:.2f}   "
                  f"Consistent {s['consistent']['mean']:6.2f}±{s['consistent']['std']:.2f}   "
                  f"ratio {s['ratio_mean']['mean']:.4f}±{s['ratio_mean']['std']:.4f}")
        m = mode_out["summary"]["ms_per_board"]
        print(f"  projection: {m['mean']:.2f}±{m['std']:.2f} ms/board")
        if mode == "lazy_exact":
            print(f"  lifetime: {mode_out['lifetime']}")
            for c, row in mode_out["per_ckpt"].items():
                print(f"    {c}: {row['machinery']}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    main()
