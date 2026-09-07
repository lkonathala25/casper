from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

ROOT = Path(__file__).resolve().parents[2]
# src_warcraft must win the name collisions (data, model, minedit, ...), so it
# is inserted last and therefore ends up first.
for p in (ROOT / "tasks/warcraft", ROOT / "tasks" / "warcraft"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data import WarcraftDataset  # noqa: E402
from edge_bridge import embed_monotone_head_into_full  # noqa: E402
from lazy_exact_layer import WarcraftEdgeConstraintLayer  # noqa: E402
from node_categories import build_node_categories  # noqa: E402


def load_backbone(ckpt, grid_size, device):
    """ResNet-18 with the monotone head widened to the full alphabet."""
    sd = torch.load(ckpt, weights_only=True, map_location=device)
    for key in ("nn_state_dict", "model_state", "model_state_dict", "state_dict"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
            break
    sd = {k: v for k, v in sd.items() if k.startswith("backbone.")}
    sd = embed_monotone_head_into_full(sd, grid_size)

    _, _, _, full_dim = build_node_categories(grid_size, alphabet="full")
    net = models.resnet18(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, full_dim)
    net.load_state_dict({k[len("backbone."):]: v for k, v in sd.items()})
    return net.to(device).eval()


def softmax_per_node(logits, var_ranges):
    out = torch.zeros_like(logits)
    off = 0
    for vr in var_ranges:
        out[:, off:off + vr] = F.softmax(logits[:, off:off + vr], dim=1)
        off += vr
    return out


def count_rings(edge_assign, N):
    """On-edges not reachable by walking from the source — a workstream 5 preview."""
    n_w = N - 1
    n_right = N * n_w
    rings = 0
    for row in edge_assign:
        on = set(int(i) for i in np.nonzero(row.numpy())[0])
        adj = {}
        for idx in on:
            if idx < n_right:
                r, c = divmod(idx, n_w)
                a, b = (r, c), (r, c + 1)
            else:
                r, c = divmod(idx - n_right, N)
                a, b = (r, c), (r + 1, c)
            adj.setdefault(a, []).append((b, idx))
            adj.setdefault(b, []).append((a, idx))
        walked, cur, prev = set(), (0, 0), None
        while cur in adj:
            nxt = [(x, i) for x, i in adj[cur] if i not in walked and x != prev]
            if not nxt:
                break
            (nxt_node, idx) = nxt[0]
            walked.add(idx)
            prev, cur = cur, nxt_node
        if len(walked) != len(on):
            rings += 1
    return rings


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dead-end receipt: run the lazy-exact layer on real network "
        "outputs and report every empty feasible set."
    )
    ap.add_argument("--ckpt-dir", type=Path, default=Path(
        "/home/lk1325/mnist_experiment/logs/warcraft_ablation_edge_aux_v2/crl_in_training"))
    ap.add_argument("--data-dir", type=Path, default=Path(
        "/home/lk1325/mnist_experiment/warcraft_crl/warcraft_shortest_path_oneskin/"
        "12x12/preprocessed_4conn"))
    ap.add_argument("--grid-size", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = args.grid_size

    node_info, var_ranges, node_order, _ = build_node_categories(N, alphabet="full")
    ds = WarcraftDataset(args.data_dir, "test", node_info, node_order)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    layer = WarcraftEdgeConstraintLayer(N, alphabet="full")
    print(f"artifact: {layer.artifact_path}")
    print(f"grid {N}x{N}  test={len(ds)}  edges={layer.num_edges}  "
          f"total_dim={layer.total_dim}  device={device}")

    ckpts = sorted(args.ckpt_dir.glob("best_run*.pt"))
    report = {}
    for ck in ckpts:
        net = load_backbone(ck, N, device)
        empty_total = boards_empty = forced = steps = rings = 0
        sites = []
        with torch.no_grad():
            for imgs, _, _ in loader:
                probs = softmax_per_node(net(imgs.to(device)), var_ranges).cpu()
                _, info = layer(probs)
                empty_total += info["empty_feasible_sets"]
                boards_empty += info["boards_with_empty_feasible"]
                forced += info["forced_steps"]
                steps += info["total_steps"]
                sites.extend(info["empty_sites"][:5])
                rings += count_rings(info["edge_assignment"], N)

        entry = {
            "empty_feasible_sets": empty_total,
            "boards_with_empty_feasible": boards_empty,
            "boards": len(ds),
            "forced_step_pct": 100.0 * forced / steps,
            "boards_with_rings": rings,
        }
        report[ck.name] = entry
        print(f"\n--- {ck.name} ---")
        print(f"  empty feasible sets        : {empty_total}")
        print(f"  boards with >=1 empty      : {boards_empty}/{len(ds)}")
        print(f"  steps with forced value    : {entry['forced_step_pct']:.1f}%")
        print(f"  boards with stray rings    : {rings}/{len(ds)}   "
              f"(workstream 5 preview, not a failure here)")
        if sites:
            print(f"  first empty sites          : {sites[:5]}")

    total_empty = sum(e["empty_feasible_sets"] for e in report.values())
    print(f"\nDEAD-END RECEIPT: {total_empty} empty feasible set(s) across "
          f"{len(ckpts)} checkpoint(s) x {len(ds)} boards")
    print(f"  lifetime counter on the layer: {layer.lifetime_empty_feasible} "
          f"empty over {layer.lifetime_boards} board-passes")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(
            {"per_ckpt": report, "total_empty_feasible_sets": total_empty,
             "artifact": str(layer.artifact_path)}, indent=2))
        print(f"Saved -> {args.out_json}")


if __name__ == "__main__":
    main()
