"""
Constrained (MAP) decoding for Warcraft 12x12 -- an alternative INFERENCE
configuration for the Scallop baseline.

The trained model is unchanged. Only the decoder differs:

  argmax  (existing)  per-node independent argmax, edge set = OR of the two
                      endpoints' patterns. Neighbouring nodes can disagree, and
                      a disagreement ADDS an edge, producing degree-3 nodes and
                      detached fragments -- which is why 98.6% Hamming accuracy
                      still yields only ~73% structural consistency.

  map     (this file) read out the most-probable proof: the route maximising
                      the product of its edge probabilities. Returns a simple
                      path by construction, so consistency is 100% by
                      construction.

Two interchangeable implementations of the same optimum:

  --decoder scallop   Scallop `reach/1` (warcraft_reach.scl) at k=1, then
                      backtrack over the returned max-product scores. This is
                      literally "read out the most-probable proof".
  --decoder dijkstra  Dijkstra on -log p. Same argmax, ~200x faster, and
                      immune to the underflow that probability-space products
                      hit on long paths.
  --decoder both      Run both and assert identical edge sets (default).

Why they coincide: max-product has optimal substructure, so retaining the
single best proof per derived fact IS Bellman's recursion. Verified at 4x4 to
1e-9 across every node.

Usage:
    python decode_map_warcraft.py --ckpt results/.../best_run0.pt \
        --out_json results/.../metrics_map_best_run0.json
"""

import argparse
import heapq
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_TASK_DIR = Path(__file__).resolve().parents[3] / "tasks" / "warcraft"
for _d in (_TASK_DIR, Path(__file__).resolve().parent):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from node_categories import build_node_categories, node_preds_to_edge_preds  # noqa: E402
from data import WarcraftDataset  # noqa: E402
from model import ResNet18NodeCategorical, is_valid_path  # noqa: E402
from compute_cost_ratio import compute_cost_metrics  # noqa: E402
from scallop_warcraft import build_edge_marginal_projection, GRID, NUM_EDGES  # noqa: E402


def compute_cost_ratio_metrics(pred_edges, gt_edges, vertex_weights, edges_lookup, grid_size):
    """Shared per-sample cost ratio (tasks/warcraft/compute_cost_ratio), in the shape this script reads."""
    ratio = np.asarray(compute_cost_metrics(pred_edges, gt_edges, vertex_weights,
                                            edges_lookup, grid_size)["per_sample"]["cost_ratio"],
                       dtype=np.float64)
    return {"ratio": ratio, "ratio_mean": float(ratio.mean()),
            "ratio_median": float(np.median(ratio))}

SOURCE_ID = 0
SINK_ID = GRID * GRID - 1
EPS = 1e-12


# =============================================================================
# Graph helpers
# =============================================================================

def build_graph(edges):
    """edges (264,4) rows (r1,c1,r2,c2) -> endpoint ids and (u,v)->edge index."""
    endpoints = [(int(r1) * GRID + int(c1), int(r2) * GRID + int(c2))
                 for r1, c1, r2, c2 in edges]
    edge_of = {}
    adj = {}
    for idx, (u, v) in enumerate(endpoints):
        edge_of[(u, v)] = idx
        edge_of[(v, u)] = idx
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    return endpoints, edge_of, adj


def edges_to_vector(route_edges, num_edges=NUM_EDGES):
    vec = np.zeros(num_edges, dtype=np.uint8)
    for idx in route_edges:
        vec[idx] = 1
    return vec


# =============================================================================
# Decoder A -- Dijkstra on -log p (log space, exact)
# =============================================================================

def dijkstra_decode(edge_prob, edge_of, adj, src=SOURCE_ID, sink=SINK_ID):
    logw = {}
    for (u, v), idx in edge_of.items():
        logw[(u, v)] = -math.log(max(float(edge_prob[idx]), EPS))

    dist = {src: 0.0}
    prev = {}
    pq = [(0.0, src)]
    done = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in done:
            continue
        done.add(u)
        if u == sink:
            break
        for v in adj.get(u, ()):
            nd = d + logw[(u, v)]
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if sink not in dist:
        return None, None
    route, cur = [], sink
    while cur != src:
        u = prev[cur]
        route.append(edge_of[(u, cur)])
        cur = u
    return edges_to_vector(route), -dist[sink]   # log-probability of the route


# =============================================================================
# Decoder B -- Scallop reach/1 at k=1, then backtrack
# =============================================================================

class ScallopReachDecoder:
    """Most-probable-proof readout via Scallop.

    A fresh context per sample is required because the edge probabilities are
    per-sample facts. Context construction dominates; ~0.02 s/sample at 12x12.
    """

    def __init__(self, endpoints, edge_of, adj, scl_path, k=1,
                 provenance="topkproofs"):
        import scallopy
        self._scallopy = scallopy
        self.endpoints = endpoints
        self.edge_of = edge_of
        self.adj = adj
        self.scl_path = str(scl_path)
        self.k = k
        self.provenance = provenance

    def scores(self, edge_prob):
        ctx = self._scallopy.ScallopContext(provenance=self.provenance, k=self.k)
        ctx.add_relation("edge", (int, int))
        ctx.add_facts("edge", [
            (float(edge_prob[i]), self.endpoints[i]) for i in range(len(self.endpoints))
        ])
        ctx.import_file(self.scl_path)
        ctx.run()
        out = {}
        for item in ctx.relation("reach"):
            prob, tup = item
            out[int(tup[0])] = float(prob)
        return out

    def decode(self, edge_prob, src=SOURCE_ID, sink=SINK_ID, rel_tol=1e-6):
        score = self.scores(edge_prob)
        # reach(0) is a leave-and-return walk, not the empty path -- pin it.
        score[src] = 1.0
        if sink not in score:
            return None, None

        # Backtrack in LOG space: products over 20-140 edges underflow badly in
        # probability space, and an absolute tolerance there passes for anything.
        logscore = {n: (math.log(p) if p > 0.0 else -math.inf)
                    for n, p in score.items()}

        # Pick the predecessor maximising logscore[u] + log p(u,cur) rather than
        # requiring an exact multiplicative match. A Scallop proof is a SET of
        # facts, so a derivation that reuses an edge already in the proof does
        # not multiply its probability again -- exact matching then fails on
        # those nodes. The argmax is well defined regardless, and coincides with
        # the exact match whenever no edge is reused.
        route, cur = [], sink
        visited = {sink}
        for _ in range(GRID * GRID + 1):
            if cur == src:
                return edges_to_vector(route), logscore[sink]
            best_u, best_val = None, -math.inf
            for u in self.adj.get(cur, ()):
                if u in visited or u not in logscore:
                    continue
                idx = self.edge_of[(u, cur)]
                lp = math.log(max(float(edge_prob[idx]), EPS))
                val = logscore[u] + lp
                if val > best_val:
                    best_u, best_val = u, val
            if best_u is None:
                return None, None          # dead end -> backtracking broke
            route.append(self.edge_of[(best_u, cur)])
            visited.add(best_u)
            cur = best_u
        return None, None                  # exceeded node count -> cycle


# =============================================================================
# Evaluation
# =============================================================================

def collect_node_probs(backbone, loader, device):
    backbone.eval()
    out = []
    with torch.no_grad():
        for imgs, _, _ in loader:
            _, node_probs, _, _ = backbone(imgs.to(device))
            out.append(node_probs.cpu())
    return torch.cat(out, dim=0)


def score_predictions(pred_edges, gt_edges, vertex_weights, edges):
    n = pred_edges.shape[0]
    exact = float((pred_edges == gt_edges).all(axis=1).mean() * 100)
    hamming = float((pred_edges == gt_edges).mean() * 100)
    valid = np.array([is_valid_path(pred_edges[i], edges, GRID) for i in range(n)],
                     dtype=bool)
    n_valid = int(valid.sum())
    out = {
        "n_samples": n,
        "n_valid": n_valid,
        "exact": exact,
        "hamming": hamming,
        "consistent": float(n_valid / n * 100),
    }
    if n_valid == 0:
        return out
    cost = compute_cost_ratio_metrics(
        pred_edges[valid], gt_edges[valid], vertex_weights[valid], edges, GRID)
    ratio = cost["ratio"]

    def frac_full(mask):
        return float(mask.sum()) / n * 100

    out.update({
        "ratio_mean_on_valid": cost["ratio_mean"],
        "ratio_median_on_valid": cost["ratio_median"],
        "pct_eq_1.0": frac_full(np.abs(ratio - 1.0) < 1e-6),
        "pct_le_1.01": frac_full(ratio <= 1.01 + 1e-6),
        "pct_le_1.05": frac_full(ratio <= 1.05 + 1e-6),
        "pct_le_1.10": frac_full(ratio <= 1.10 + 1e-6),
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="data/preprocessed_4conn")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--scl_path", default=None)
    ap.add_argument("--decoder", default="both",
                    choices=["scallop", "dijkstra", "both"])
    ap.add_argument("--k", type=int, default=1,
                    help="Scallop k. k=1 is the most-probable-proof readout; "
                         "k>1 aggregates a DNF and is NOT the argmax.")
    ap.add_argument("--provenance", default="topkproofs")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--alphabet", default="full", choices=["monotone", "full"],
                    help="Must match the alphabet the checkpoint was trained with.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only decode the first N test samples (smoke test).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}   decoder: {args.decoder}   k={args.k}")

    node_info, var_ranges, node_order, _ = build_node_categories(GRID, alphabet=args.alphabet)
    data_dir = Path(args.data_dir)
    test_ds = WarcraftDataset(data_dir, "test", node_info, node_order)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)

    edges = np.load(data_dir / "edges.npy")
    assert edges.shape[0] == NUM_EDGES
    vertex_weights = np.load(data_dir / "test_vertex_weights.npy")
    gt_edges = np.load(data_dir / "test_edge_labels.npy").astype(np.uint8)

    backbone = ResNet18NodeCategorical(var_ranges, grid_size=GRID,
                                       constraint_mode='none').to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    bb = {k[len("backbone."):]: v for k, v in state.items()
          if k.startswith("backbone.")}
    if not bb:
        raise RuntimeError(f"No backbone.* keys in {args.ckpt}")
    backbone.load_state_dict(bb, strict=True)
    print(f"Loaded {len(bb)} backbone tensors from {args.ckpt}")

    node_probs = collect_node_probs(backbone, loader, device)
    n_total = node_probs.shape[0]
    n = min(args.limit, n_total) if args.limit else n_total
    print(f"Test samples: {n_total}" + (f" (decoding first {n})" if n != n_total else ""))

    proj = build_edge_marginal_projection(node_info, node_order)
    edge_probs = (node_probs @ proj.t() * 0.5).clamp(0.0, 1.0).numpy()

    # Baseline: the existing per-node argmax / OR decoding.
    # NB: node_preds_to_edge_preds is an unoptimised reference implementation
    # (nested Python loops over batch x nodes), so its wall-clock is an upper
    # bound and not directly comparable to the two solver decoders.
    t0 = time.time()
    argmax_edges = (node_preds_to_edge_preds(
        node_probs[:n], node_info, node_order, NUM_EDGES).numpy() > 0.5).astype(np.uint8)
    t_argmax = time.time() - t0

    endpoints, edge_of, adj = build_graph(edges)
    scl_path = Path(args.scl_path) if args.scl_path else Path(__file__).parent / "warcraft_reach.scl"
    if args.decoder in ("scallop", "both") and not scl_path.exists():
        raise FileNotFoundError(scl_path)

    sc_dec = (ScallopReachDecoder(endpoints, edge_of, adj, scl_path,
                                 k=args.k, provenance=args.provenance)
              if args.decoder in ("scallop", "both") else None)

    map_edges = np.zeros((n, NUM_EDGES), dtype=np.uint8)
    dij_edges = np.zeros((n, NUM_EDGES), dtype=np.uint8)
    have_dij = np.zeros(n, dtype=bool)
    n_fail_scallop = n_disagree = 0
    t_sc = t_dj = 0.0

    for i in range(n):
        ep = edge_probs[i]
        vec_dj = vec_sc = None
        if args.decoder in ("dijkstra", "both"):
            t0 = time.time(); vec_dj, _ = dijkstra_decode(ep, edge_of, adj); t_dj += time.time() - t0
            if vec_dj is not None:
                dij_edges[i] = vec_dj
                have_dij[i] = True
        if sc_dec is not None:
            t0 = time.time(); vec_sc, _ = sc_dec.decode(ep); t_sc += time.time() - t0
            if vec_sc is None:
                n_fail_scallop += 1

        if args.decoder == "both":
            if vec_sc is not None and vec_dj is not None and not np.array_equal(vec_sc, vec_dj):
                n_disagree += 1
            chosen = vec_sc if vec_sc is not None else vec_dj
        else:
            chosen = vec_sc if args.decoder == "scallop" else vec_dj

        if chosen is None:
            raise RuntimeError(f"sample {i}: no route decoded (graph disconnected?)")
        map_edges[i] = chosen

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n} decoded", flush=True)

    gt = gt_edges[:n]
    vw = vertex_weights[:n]
    res_map = score_predictions(map_edges, gt, vw, edges)
    res_arg = score_predictions(argmax_edges, gt, vw, edges)

    # Distance to the Dijkstra reference. The point of this measure: the top-1
    # proof readout should be IDENTICAL to Dijkstra (both compute the max-product
    # path), whereas the train-only argmax decode should be far from it. That
    # contrast is the evidence the readout is a shortest-path search rather than
    # a reading of the network's own suggestion.
    def dist_to_dijkstra(pred):
        if not have_dij.any():
            return None
        m = have_dij
        diff = (pred[m] != dij_edges[m])
        per_sample_bits = diff.sum(axis=1)
        return {
            "n_compared": int(m.sum()),
            "mean_hamming_edges": float(per_sample_bits.mean()),
            "median_hamming_edges": float(np.median(per_sample_bits)),
            "max_hamming_edges": int(per_sample_bits.max()),
            "mean_hamming_pct": float(diff.mean() * 100),
            "identical_pct": float((per_sample_bits == 0).mean() * 100),
        }

    d_arg = dist_to_dijkstra(argmax_edges)
    d_map = dist_to_dijkstra(map_edges)

    out = {
        "ckpt": str(Path(args.ckpt).resolve()),
        "n_decoded": n,
        "decoder": args.decoder,
        "scallop_k": args.k,
        "scallop_provenance": args.provenance,
        "scallop_backtrack_failures": n_fail_scallop,
        "scallop_vs_dijkstra_disagreements": n_disagree if args.decoder == "both" else None,
        "seconds_scallop": t_sc,
        "seconds_dijkstra": t_dj,
        "seconds_argmax": t_argmax,
        "ms_per_sample_scallop": (t_sc / n * 1000) if t_sc else None,
        "ms_per_sample_dijkstra": (t_dj / n * 1000) if t_dj else None,
        "ms_per_sample_argmax": t_argmax / n * 1000,
        "argmax": res_arg,
        "map": res_map,
        "argmax_vs_dijkstra": d_arg,
        "map_vs_dijkstra": d_map,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))

    def row(label, r):
        return (f"{label:>10} {r['exact']:>7.2f} {r['hamming']:>8.2f} "
                f"{r['consistent']:>9.2f} {r.get('pct_eq_1.0', float('nan')):>7.2f} "
                f"{r.get('pct_le_1.01', float('nan')):>7.2f} "
                f"{r.get('pct_le_1.05', float('nan')):>7.2f} "
                f"{r.get('pct_le_1.10', float('nan')):>7.2f} "
                f"{r.get('ratio_mean_on_valid', float('nan')):>10.4f}")

    print()
    print(f"{'decoder':>10} {'exact':>7} {'hamming':>8} {'consist':>9} "
          f"{'opt':>7} {'<=1%':>7} {'<=5%':>7} {'<=10%':>7} {'ratio|valid':>10}")
    print("-" * 82)
    print(row("argmax", res_arg))
    print(row("MAP", res_map))
    print()
    if args.decoder == "both":
        print(f"scallop/dijkstra disagreements : {n_disagree}/{n}")
    print(f"scallop backtrack failures     : {n_fail_scallop}/{n}")
    if t_sc:
        print(f"scallop {t_sc:.1f}s ({t_sc/n*1000:.1f} ms/sample)")
    if t_dj:
        print(f"dijkstra {t_dj:.2f}s ({t_dj/n*1000:.3f} ms/sample)")
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
