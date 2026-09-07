import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
for _d in (_ROOT / "casper", Path(__file__).resolve().parent):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from node_categories import build_node_categories  # noqa: E402
from steps_layer import StepsProjectionLayer  # noqa: E402
from steps_walk import edge_index_maps  # noqa: E402


def path_moves(edge_vec, N):
    """Ordered (dr, dc) moves along the s-t path carried by a binary edge vector."""
    edges, _, _ = edge_index_maps(N)
    adj = {}
    for e, on in enumerate(edge_vec):
        if not on:
            continue
        r1, c1, r2, c2 = edges[e]
        a, b = r1 * N + c1, r2 * N + c2
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seq, prev, cur = [0], None, 0
    target = N * N - 1
    while cur != target:
        nxt = [v for v in adj.get(cur, []) if v != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        seq.append(cur)
    moves = []
    for a, b in zip(seq, seq[1:]):
        moves.append((b // N - a // N, b % N - a % N))
    return moves


def is_non_monotone(edge_vec, N):
    """True iff the path takes at least one up or left move."""
    return any(dr < 0 or dc < 0 for dr, dc in path_moves(edge_vec, N))


def main() -> None:
    """Prove the steps pipeline represents and reproduces non-monotone paths."""
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--grid-size", type=int, default=12)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--artifact", type=Path, default=None,
                   help="Compiled steps artifact to drive the layer. Defaults "
                        "to the one for this grid size; pass --no-artifact to "
                        "test the generator path instead.")
    p.add_argument("--no-artifact", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    N = args.grid_size
    edge_labels = np.load(args.data_dir / f"{args.split}_edge_labels.npy")
    edges_lookup = np.load(args.data_dir / "edges.npy")
    if args.limit:
        edge_labels = edge_labels[: args.limit]
    B = edge_labels.shape[0]

    non_mono = np.array([is_non_monotone(edge_labels[b], N) for b in range(B)])
    result = {
        "grid": N,
        "split": args.split,
        "boards": B,
        "gt_non_monotone": int(non_mono.sum()),
        "gt_non_monotone_pct": round(100.0 * float(non_mono.mean()), 2),
    }

    for alphabet in ("monotone", "full"):
        info, _, order, dim = build_node_categories(N, alphabet=alphabet)
        result[f"alphabet_{alphabet}_dim"] = dim

    art = None
    if not args.no_artifact:
        art = args.artifact or (
            _ROOT / "constraints" / "warcraft" / "steps_artifacts"
            / f"steps_{N}_trimmed_compiled.json"
        )
        if not art.exists():
            raise FileNotFoundError(f"no compiled artifact at {art}")
    layer = StepsProjectionLayer(grid_size=N, alphabet="full", certify=True,
                                 compiled_artifact=art)
    result["driven_by"] = "compiled_artifact" if art else "generator"
    scores = layer.score_bridge.gt_cell_scores(edge_labels, edges_lookup)
    edge_assign, _ = layer.project_edges(scores)

    exact = np.array([bool((edge_assign[b] == edge_labels[b]).all())
                      for b in range(B)])
    emitted_non_mono = np.array([is_non_monotone(edge_assign[b], N)
                                 for b in range(B)])
    result["passthrough_exact_all"] = round(100.0 * float(exact.mean()), 2)
    result["passthrough_exact_on_non_monotone"] = (
        round(100.0 * float(exact[non_mono].mean()), 2) if non_mono.any() else None
    )
    result["layer_emitted_non_monotone"] = int(emitted_non_mono.sum())

    verdict = (
        result["gt_non_monotone"] > 0
        and result["passthrough_exact_on_non_monotone"] == 100.0
        and result["layer_emitted_non_monotone"] == result["gt_non_monotone"]
    )
    result["PASS"] = verdict

    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
