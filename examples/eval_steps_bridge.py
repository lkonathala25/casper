import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
for sub in ("tasks/warcraft", "tasks/warcraft"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from compute_cost_ratio import _is_valid_src_sink_path  # noqa: E402
from edge_bridge import embed_monotone_head_into_full  # noqa: E402

from steps_bridge import PlainBroadcastBridge, softmax_per_node  # noqa: E402
from steps_walk import StepsWalkDecoder, edge_endpoints_array  # noqa: E402

INFERENCE_ONLY_NOTE = (
    "the tex specifies no training regime; inference-only is our choice, "
    "matching the lazy-exact evaluation"
)
BRIDGE_NOTE = (
    "bridge = plain broadcast, assumed equivalent to the reference lambda=0 "
    "— unverified assumption."
)


def edges_lookup_for(N):
    """(n_edges, 4) rows (r1, c1, r2, c2) under the canonical edge indexing."""
    n_w = N - 1
    lut = np.zeros((2 * N * n_w, 4), dtype=np.int64)
    for i in range(N):
        for j in range(n_w):
            lut[i * n_w + j] = (i, j, i, j + 1)
    for i in range(n_w):
        for j in range(N):
            lut[N * n_w + i * N + j] = (i, j, i + 1, j)
    return lut


def load_backbone(ckpt, total_dim, N, device):
    """Load the ResNet-18 categorical head, widening a monotone head if needed.

    Same path the lazy-exact evaluation takes, so the two pipelines read the
    identical network outputs.
    """
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


def run_pipeline(dec, bridge, cell_scores, endpoints, src_flat, sink_flat, gt=None,
                 tie_break="no-backtrack"):
    """bridge -> walk -> early cut -> loop erasure -> certificate, per board."""
    L = dec.enc.horizon
    rows = []
    for b in range(cell_scores.shape[0]):
        ticks = bridge.tick_scores(cell_scores[b], L)
        t0 = time.perf_counter()
        seq, stats = dec.walk(
            ticks, selection="argmax", early_cut=True, tie_break=tie_break
        )
        erased, snipped, vec = dec.decode(seq)
        ms = (time.perf_counter() - t0) * 1e3
        ok = _is_valid_src_sink_path(vec.astype(bool), endpoints, src_flat, sink_flat)
        row = {
            "board": b,
            "arrival": stats.arrival,
            "dead_ends": stats.dead_ends,
            "decisions": stats.decisions,
            "snipped": snipped,
            "valid": bool(ok),
            "ms": ms,
            "erased": erased,
            "edges": vec,
        }
        if gt is not None:
            row["exact_match"] = bool(np.array_equal(vec.astype(bool), gt[b].astype(bool)))
        rows.append(row)
    return rows


def summarise(rows, label):
    """Aggregate the per-board rows into the gate's reported numbers."""
    n = len(rows)
    snipped = [r["snipped"] for r in rows]
    out = {
        "label": label,
        "boards": n,
        "certificates_passed": sum(r["valid"] for r in rows),
        "certificate_failures": n - sum(r["valid"] for r in rows),
        "dead_ends": sum(r["dead_ends"] for r in rows),
        "boards_needing_erasure": sum(1 for s in snipped if s),
        "pct_boards_needing_erasure": 100.0 * sum(1 for s in snipped if s) / n,
        "mean_cells_snipped": float(np.mean(snipped)),
        "mean_arrival": float(np.mean([r["arrival"] for r in rows])),
        "ms_per_board": float(np.mean([r["ms"] for r in rows])),
    }
    if "exact_match" in rows[0]:
        em = sum(r["exact_match"] for r in rows)
        out["exact_match"] = em
        out["exact_match_pct"] = 100.0 * em / n
    return out


def main() -> None:
    """Run the Phase 4 bridge gates: GT pass-through, checkpoint smoke, determinism."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-size", type=int, default=12)
    ap.add_argument("--data-dir", type=Path, default=Path(
        "/home/lk1325/mnist_experiment/warcraft_crl/warcraft_shortest_path_oneskin/"
        "12x12/preprocessed_4conn"))
    ap.add_argument("--ckpt", type=Path, default=Path(
        "/home/lk1325/mnist_experiment/logs/warcraft_ablation_edge_aux_v2/"
        "crl_in_training/best_run0.pt"))
    ap.add_argument("--smoke-boards", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "logs" / "steps_phase4_bridge.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    N = args.grid_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{INFERENCE_ONLY_NOTE}\n{BRIDGE_NOTE}\ndevice={device}", flush=True)

    bridge = PlainBroadcastBridge(N, alphabet="full")
    dec = StepsWalkDecoder(N)
    endpoints = edge_endpoints_array(N)
    src_flat, sink_flat = 0, (N - 1) * N + (N - 1)
    lut = edges_lookup_for(N)

    gt = np.load(args.data_dir / "test_edge_labels.npy").astype(np.uint8)
    maps = np.load(args.data_dir / "test_maps.npy")
    print(f"test boards={len(gt)}  total_dim={bridge.total_dim}", flush=True)

    result = {
        "grid": N,
        "seed": args.seed,
        "inference_only_note": INFERENCE_ONLY_NOTE,
        "bridge_note": BRIDGE_NOTE,
        "tie_break": "no-backtrack (adopted)",
        "on_path_scalar": "P(on-path) = 1 - P(skip); source/sink have no skip "
                          "category so score 1.0",
        "ckpt": str(args.ckpt),
        "data_dir": str(args.data_dir),
    }

    # Gate 1 — GT pass-through on every test board.
    gt_scores = bridge.gt_cell_scores(gt, lut)
    rows = run_pipeline(dec, bridge, gt_scores, endpoints, src_flat, sink_flat, gt=gt)
    result["gt_passthrough"] = summarise(rows, "gt_passthrough")
    print(json.dumps(result["gt_passthrough"], indent=2), flush=True)
    misses = [r["board"] for r in rows if not r["exact_match"]][:5]
    if misses:
        result["gt_passthrough"]["first_misses"] = misses
        for b in misses[:2]:
            r = rows[b]
            gt_cells = sorted(np.nonzero(gt_scores[b])[0].tolist())
            result["gt_passthrough"].setdefault("miss_examples", []).append(
                {"board": b, "walk_erased": r["erased"], "gt_on_path_cells": gt_cells}
            )

    # Retained as the before picture for the adopted tie-break.
    diag = run_pipeline(
        dec, bridge, gt_scores, endpoints, src_flat, sink_flat, gt=gt,
        tie_break="index",
    )
    result["gt_passthrough_index_tiebreak_before"] = summarise(
        diag, "gt_passthrough_index_tiebreak_before"
    )
    print(
        json.dumps(result["gt_passthrough_index_tiebreak_before"], indent=2),
        flush=True,
    )

    # Gate 2 — real checkpoint smoke.
    net = load_backbone(args.ckpt, bridge.total_dim, N, device)
    imgs = torch.from_numpy(maps[: args.smoke_boards]).float().div_(255.0)
    imgs = imgs.permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        probs = softmax_per_node(net(imgs), bridge.var_ranges)
    ckpt_scores = bridge.cell_scores(probs)
    rows_ck = run_pipeline(
        dec, bridge, ckpt_scores, endpoints, src_flat, sink_flat, gt=gt
    )
    result["checkpoint_smoke"] = summarise(rows_ck, "checkpoint_smoke")
    result["checkpoint_smoke_index_tiebreak_before"] = summarise(
        run_pipeline(
            dec, bridge, ckpt_scores, endpoints, src_flat, sink_flat, gt=gt,
            tie_break="index",
        ),
        "checkpoint_smoke_index_tiebreak_before",
    )
    result["checkpoint_smoke"]["examples"] = [
        {
            "board": r["board"],
            "arrival": r["arrival"],
            "snipped": r["snipped"],
            "valid": r["valid"],
            "walk_cells": r["erased"],
            "edge_indices": np.nonzero(r["edges"])[0].tolist(),
        }
        for r in rows_ck[:3]
    ]
    print(json.dumps(result["checkpoint_smoke"], indent=2), flush=True)

    # The walk softmaxes over D_i before selecting, purely so MinEditL2 sees a
    # distribution. Softmax is strictly monotone, so it must not change any
    # argmax — i.e. the bridge really does apply no normalization.
    mono_ok = True
    for b in range(min(20, ckpt_scores.shape[0])):
        ticks = bridge.tick_scores(ckpt_scores[b], dec.enc.horizon)
        prev = dec.enc.source
        for i in range(1, dec.enc.horizon + 1):
            cands = dec.feasible(i, prev)
            if len(cands) > 1:
                dom = dec.enc.domains[i]
                row = torch.as_tensor([ticks[i][v] for v in dom], dtype=torch.float32)
                sm = torch.softmax(row, dim=0)
                raw_pick = max(cands, key=lambda c: float(ticks[i][c]))
                sm_pick = max(cands, key=lambda c: float(sm[dec.local[i][c]]))
                if raw_pick != sm_pick:
                    mono_ok = False
                prev = sm_pick
            else:
                prev = cands[0]
    result["no_hidden_normalization"] = {
        "boards_checked": min(20, ckpt_scores.shape[0]),
        "raw_argmax_equals_softmax_argmax": mono_ok,
    }
    print(json.dumps(result["no_hidden_normalization"], indent=2), flush=True)

    # Gate 3 — determinism.
    a = run_pipeline(dec, bridge, ckpt_scores[:5], endpoints, src_flat, sink_flat)
    b = run_pipeline(dec, bridge, ckpt_scores[:5], endpoints, src_flat, sink_flat)
    result["determinism"] = {
        "boards": 5,
        "identical": all(
            x["erased"] == y["erased"] and np.array_equal(x["edges"], y["edges"])
            for x, y in zip(a, b)
        ),
    }
    print(json.dumps(result["determinism"], indent=2), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
