from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "casper", ROOT / "tasks" / "warcraft"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from constraints import Constraints  # noqa: E402
from edge_bridge import CategoricalEdgeBridge, default_artifact  # noqa: E402
from lazy_exact_layer import WarcraftLazyExactLayer  # noqa: E402
from node_categories import (  # noqa: E402
    build_node_categories,
    edge_labels_to_node_labels,
    node_preds_to_edge_preds,
)


def edge_vector(path, N):
    n_w, n_right = N - 1, N * (N - 1)
    vec = np.zeros(2 * N * (N - 1), dtype=np.int64)
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if r1 == r2:
            vec[r1 * n_w + min(c1, c2)] = 1
        else:
            vec[n_right + min(r1, r2) * N + c1] = 1
    return vec


def non_monotone_path(N):
    """Detours up and left out of the corner, then runs to the sink."""
    return ([(0, 0), (1, 0), (1, 1), (0, 1)]
            + [(0, c) for c in range(2, N)]
            + [(r, N - 1) for r in range(1, N)])


def onehot(labels, var_ranges, total_dim):
    B = labels.shape[0]
    oh = torch.zeros(B, total_dim)
    off = 0
    for k, vr in enumerate(var_ranges):
        oh[torch.arange(B), off + labels[:, k]] = 1.0
        off += vr
    return oh


def sample_valid(cons, n_edges, n, seed=0):
    ordering = list(cons.ordering)
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        a, ok = {}, True
        for step, var in enumerate(ordering):
            fs = sorted(cons.feasible_set(step, a))
            if not fs:
                ok = False
                break
            a[var] = rng.choice(fs) if len(fs) > 1 else fs[0]
        if ok:
            out.append([a[v] for v in range(n_edges)])
    return np.array(out, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pre-evaluation gates for the lazy-exact layer at a grid size.")
    ap.add_argument("--grid-size", type=int, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--n-random", type=int, default=100)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    N = args.grid_size
    n_edges = 2 * N * (N - 1)
    report = {"grid": N, "edges": n_edges}
    ok_all = True

    art = default_artifact(N)
    t0 = time.perf_counter()
    cons = Constraints.from_compiled_dsl(art)
    load_s = time.perf_counter() - t0
    n_clauses = sum(len(p) for p in cons._impl.compiled.per_var_clauses)
    print(f"artifact : {art}")
    print(f"  variables={len(cons.ordering)}  clauses={n_clauses}  "
          f"load={load_s:.3f}s")
    report.update({"artifact": str(art), "variables": len(cons.ordering),
                   "clauses": n_clauses, "load_s": load_s})

    layer = WarcraftLazyExactLayer(N, alphabet="full")
    bridge = CategoricalEdgeBridge(N, alphabet="full")
    node_info, var_ranges, node_order, total_dim = build_node_categories(
        N, alphabet="full")
    report["total_dim"] = total_dim

    # (b) bridge round trip on random degree-legal edge sets
    t0 = time.perf_counter()
    samples = sample_valid(cons, n_edges, args.n_random, seed=0)
    labels = bridge.edges_to_categories(torch.from_numpy(samples))
    recovered = bridge.categories_to_edges(torch.from_numpy(labels)).numpy()
    rt_ok = int((recovered == samples).all(axis=1).sum())
    again = bridge.edges_to_categories(torch.from_numpy(recovered))
    rt2_ok = bool((again == labels).all())
    dt = time.perf_counter() - t0
    print(f"\n(b) bridge round trip on {len(samples)} random degree-legal sets: "
          f"edge->cat->edge {rt_ok}/{len(samples)}, cat->edge->cat {rt2_ok} "
          f"[{dt:.1f}s]")
    ok_all &= (rt_ok == len(samples)) and rt2_ok
    report["bridge_roundtrip"] = {"n": len(samples), "edge_cat_edge_ok": rt_ok,
                                  "cat_edge_cat_ok": rt2_ok}

    # (c) a known non-monotone path passes unedited
    p = non_monotone_path(N)
    assert len(set(p)) == len(p) and p[0] == (0, 0) and p[-1] == (N - 1, N - 1)
    vec = edge_vector(p, N)[None, :]
    lab = bridge.edges_to_categories(torch.from_numpy(vec))
    probs = onehot(torch.from_numpy(lab), var_ranges, total_dim)
    _, info = layer(probs)
    nm_ok = (info["cells_edited"] == 0
             and (info["edge_assignment"].numpy() == vec).all()
             and info["empty_feasible_sets"] == 0)
    print(f"(c) non-monotone path passes unedited: {nm_ok} "
          f"(cells_edited={info['cells_edited']}, "
          f"rounds={info['per_board'][0]['rounds']})")
    ok_all &= bool(nm_ok)
    report["non_monotone_unedited"] = bool(nm_ok)

    # (a) GT passthrough
    gt = np.load(args.data_dir / "test_edge_labels.npy").astype(np.int64)
    t0 = time.perf_counter()
    gt_labels = edge_labels_to_node_labels(gt, node_info, node_order)
    oh = onehot(torch.from_numpy(gt_labels), var_ranges, total_dim)
    preds = []
    for s in range(0, len(gt), args.batch_size):
        proj, _ = layer(oh[s:s + args.batch_size])
        e = node_preds_to_edge_preds(proj, node_info, node_order, n_edges)
        preds.append((e > 0.5).numpy().astype(np.int64))
    pred = np.concatenate(preds)
    em = float((pred == gt).all(axis=1).mean() * 100)
    ham = float((pred == gt).mean() * 100)
    dt = time.perf_counter() - t0
    print(f"(a) GT passthrough over {len(gt)} paths: EM={em:.2f} "
          f"Hamming={ham:.4f} [{dt:.1f}s]")
    ok_all &= (em == 100.0)
    report["gt_passthrough"] = {"n": len(gt), "exact": em, "hamming": ham}

    report["dead_end_lifetime"] = {
        "empty_feasible": layer.lifetime_empty_feasible,
        "boards": layer.lifetime_boards,
        "capped": layer.lifetime_capped_boards,
        "rings_forbidden": layer.lifetime_rings_forbidden,
    }
    print(f"\ndead-end counter: {report['dead_end_lifetime']}")
    print(f"GATES: {'PASS' if ok_all else 'FAIL'}")
    report["gates_pass"] = bool(ok_all)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2))
        print(f"Saved -> {args.out_json}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
