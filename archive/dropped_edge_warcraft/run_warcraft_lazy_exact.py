from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

ROOT = Path(__file__).resolve().parent.parent
# src_warcraft must win the name collisions (data, model, minedit, ...).
for p in (ROOT / "tasks/warcraft", ROOT / "tasks" / "warcraft"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data import WarcraftDataset  # noqa: E402
from edge_bridge import embed_monotone_head_into_full  # noqa: E402
from lazy_exact_layer import WarcraftLazyExactLayer  # noqa: E402
from node_categories import build_node_categories  # noqa: E402


def load_backbone(ckpt, grid_size, device):
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the lazy-exact ring-elimination loop over the test set "
        "and report per-board rounds, rings and flips."
    )
    ap.add_argument("--ckpt-dir", type=Path, default=Path(
        "/home/lk1325/mnist_experiment/logs/warcraft_ablation_edge_aux_v2/crl_in_training"))
    ap.add_argument("--data-dir", type=Path, default=Path(
        "/home/lk1325/mnist_experiment/warcraft_crl/warcraft_shortest_path_oneskin/"
        "12x12/preprocessed_4conn"))
    ap.add_argument("--grid-size", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=250)
    ap.add_argument("--max-rounds", type=int, default=20)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = args.grid_size

    node_info, var_ranges, node_order, _ = build_node_categories(N, alphabet="full")
    ds = WarcraftDataset(args.data_dir, "test", node_info, node_order)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    layer = WarcraftLazyExactLayer(N, alphabet="full", max_rounds=args.max_rounds)
    print(f"artifact  : {layer.artifact_path}")
    print(f"grid {N}x{N}  test={len(ds)}  max_rounds={args.max_rounds}  device={device}")

    report = {}
    for ck in sorted(args.ckpt_dir.glob("best_run*.pt")):
        net = load_backbone(ck, N, device)
        rounds_hist = Counter()
        rings_round1 = 0
        total_rings = 0
        flips = []
        capped = []
        violating = []
        empty_total = 0
        certified = 0
        n = 0

        with torch.no_grad():
            for imgs, _, _ in loader:
                probs = softmax_per_node(net(imgs.to(device)), var_ranges).cpu()
                _, info = layer(probs)
                for lg in info["per_board"]:
                    rounds_hist[lg["rounds"]] += 1
                    rings_round1 += lg["rings_per_round"][0]
                    total_rings += sum(lg["rings_per_round"])
                    flips.append(sum(lg["edges_flipped_per_round"]))
                    certified += 1 if lg["certified"] else 0
                    if lg["capped"]:
                        capped.append(n)
                    if lg["violations"]:
                        violating.append((n, lg["violations"]))
                    n += 1
                empty_total += info["empty_feasible_sets"]

        refined = sum(c for r, c in rounds_hist.items() if r > 1)
        entry = {
            "boards": n,
            "certified": certified,
            "certified_pct": 100.0 * certified / n,
            "rounds_histogram": {str(k): v for k, v in sorted(rounds_hist.items())},
            "boards_certified_round1": rounds_hist.get(1, 0),
            "boards_needing_refinement": refined,
            "refinement_pct": 100.0 * refined / n,
            "max_rounds_used": max(rounds_hist) if rounds_hist else 0,
            "rings_found_round1": rings_round1,
            "rings_found_total": total_rings,
            "mean_edges_flipped_on_refined": (
                sum(f for f in flips if f) / max(refined, 1)),
            "max_edges_flipped": max(flips) if flips else 0,
            "capped_boards": capped,
            "validity_violations": violating,
            "empty_feasible_sets": empty_total,
        }
        report[ck.name] = entry

        print(f"\n--- {ck.name} ---")
        print(f"  certified                 : {certified}/{n} "
              f"({entry['certified_pct']:.1f}%)")
        print(f"  certified in round 1      : {entry['boards_certified_round1']} "
              f"({100.0 * entry['boards_certified_round1'] / n:.1f}%)")
        print(f"  needed refinement         : {refined} ({entry['refinement_pct']:.1f}%)")
        print(f"  rounds histogram          : {entry['rounds_histogram']}")
        print(f"  rings found (round 1/all) : {rings_round1} / {total_rings}")
        print(f"  mean edges flipped        : "
              f"{entry['mean_edges_flipped_on_refined']:.1f} "
              f"(max {entry['max_edges_flipped']})")
        print(f"  capped boards             : {len(capped)}")
        print(f"  validity violations       : {len(violating)}")
        print(f"  empty feasible sets       : {empty_total}")

    print(f"\nLIFETIME on the layer: empty_feasible={layer.lifetime_empty_feasible} "
          f"over {layer.lifetime_boards} board-passes; "
          f"capped={layer.lifetime_capped_boards}; "
          f"rings_forbidden={layer.lifetime_rings_forbidden}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({
            "per_ckpt": report,
            "lifetime": {
                "empty_feasible": layer.lifetime_empty_feasible,
                "boards": layer.lifetime_boards,
                "capped": layer.lifetime_capped_boards,
                "rings_forbidden": layer.lifetime_rings_forbidden,
            },
            "artifact": str(layer.artifact_path),
            "max_rounds": args.max_rounds,
        }, indent=2))
        print(f"Saved -> {args.out_json}")


if __name__ == "__main__":
    main()
