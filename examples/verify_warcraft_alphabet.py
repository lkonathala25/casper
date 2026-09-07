import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tasks" / "warcraft"))

from node_categories import (  # noqa: E402
    UnrepresentablePatternError,
    build_node_categories,
    edge_labels_to_node_labels,
    node_preds_to_edge_preds,
)


def legacy_build(grid_size):
    """The pre-fix encoding, reproduced verbatim as a regression reference."""
    n_w = grid_size - 1
    n_right = grid_size * n_w

    def right_idx(i, j):
        return i * n_w + j

    def down_idx(i, j):
        return n_right + i * grid_size + j

    node_info, node_order, var_ranges = {}, [], []
    for i in range(grid_size):
        for j in range(grid_size):
            node = (i, j)
            node_order.append(node)
            incoming, outgoing = {}, {}
            if j > 0:
                incoming['R'] = right_idx(i, j - 1)
            if i > 0:
                incoming['D'] = down_idx(i - 1, j)
            if j < grid_size - 1:
                outgoing['R'] = right_idx(i, j)
            if i < grid_size - 1:
                outgoing['D'] = down_idx(i, j)
            is_source = (i == 0 and j == 0)
            is_sink = (i == grid_size - 1 and j == grid_size - 1)
            all_edges = list(incoming.values()) + list(outgoing.values())
            categories, edge_patterns = [], []
            if is_source:
                for out_label, out_idx in outgoing.items():
                    categories.append(out_label)
                    p = {e: 0 for e in all_edges}
                    p[out_idx] = 1
                    edge_patterns.append(p)
            elif is_sink:
                for in_label, in_idx in incoming.items():
                    categories.append(in_label)
                    p = {e: 0 for e in all_edges}
                    p[in_idx] = 1
                    edge_patterns.append(p)
            else:
                categories.append('skip')
                edge_patterns.append({e: 0 for e in all_edges})
                for in_label, in_idx in incoming.items():
                    for out_label, out_idx in outgoing.items():
                        categories.append(f'{in_label}->{out_label}')
                        p = {e: 0 for e in all_edges}
                        p[in_idx] = 1
                        p[out_idx] = 1
                        edge_patterns.append(p)
            node_info[node] = {'categories': categories, 'n_cats': len(categories),
                               'incoming': incoming, 'outgoing': outgoing,
                               'edge_patterns': edge_patterns, 'all_edges': all_edges,
                               'is_source': is_source, 'is_sink': is_sink}
            var_ranges.append(len(categories))
    return node_info, var_ranges, node_order, sum(var_ranges)


def legacy_convert(edge_labels, node_info, node_order):
    """The pre-fix conversion, including the silent fallback-to-category-0."""
    el = np.asarray(edge_labels).astype(int)
    node_labels = np.zeros((el.shape[0], len(node_order)), dtype=np.int64)
    for n_idx, node in enumerate(node_order):
        info = node_info[node]
        for b in range(el.shape[0]):
            observed = {e: int(el[b, e]) for e in info['all_edges']}
            for cat_idx, pattern in enumerate(info['edge_patterns']):
                if pattern == observed:
                    node_labels[b, n_idx] = cat_idx
                    break
            else:
                node_labels[b, n_idx] = 0
    return node_labels


def check_prefix_extension(N):
    """The monotone alphabet must be a strict prefix of the full one at every node."""
    mi, mvr, mo, _ = build_node_categories(N, alphabet='monotone')
    fi, fvr, fo, _ = build_node_categories(N, alphabet='full')
    assert mo == fo
    for node in mo:
        m, f = mi[node], fi[node]
        if m['edge_patterns'] != f['edge_patterns'][:m['n_cats']]:
            return False, node
    return True, None


def check_legacy_regression(N):
    """The 'monotone' alphabet must reproduce the pre-fix encoding exactly."""
    li, lvr, lo, ltd = legacy_build(N)
    ni, vr, no, td = build_node_categories(N, alphabet='monotone')
    if (lvr, lo, ltd) != (vr, no, td):
        return False, "var_ranges/node_order/total_dim differ"
    for node in lo:
        if li[node]['edge_patterns'] != ni[node]['edge_patterns']:
            return False, f"edge_patterns differ at {node}"
        if li[node]['all_edges'] != ni[node]['all_edges']:
            return False, f"all_edges differ at {node}"
    return True, None


def onehot(node_labels, var_ranges):
    import torch
    B = node_labels.shape[0]
    out = torch.zeros(B, sum(var_ranges))
    off = 0
    for k, vr in enumerate(var_ranges):
        out[torch.arange(B), off + node_labels[:, k]] = 1.0
        off += vr
    return out


def run_split(N, split, data_root, report):
    import torch
    P = Path(data_root) / f"{N}x{N}" / "preprocessed_4conn"
    el = np.load(P / f"{split}_edge_labels.npy").astype(int)
    n = el.shape[0]

    mi, mvr, mo, mtd = build_node_categories(N, alphabet='monotone')
    fi, fvr, fo, ftd = build_node_categories(N, alphabet='full')

    # Old alphabet: how many instances does the loud failure now reject?
    rejected = []
    for b in range(n):
        try:
            edge_labels_to_node_labels(el[b:b + 1], mi, mo, instance_ids=[b])
        except UnrepresentablePatternError:
            rejected.append(b)

    # New alphabet: must convert every instance without raising.
    failures = []
    try:
        new_labels = edge_labels_to_node_labels(el, fi, fo)
    except UnrepresentablePatternError as e:
        failures.append(str(e))
        new_labels = None

    entry = {
        "grid": N, "split": split, "n": n,
        "rejected_by_monotone_alphabet": len(rejected),
        "rejected_pct": 100.0 * len(rejected) / n,
        "full_alphabet_failures": len(failures),
    }

    if new_labels is not None:
        old_labels = legacy_convert(el, mi, mo)
        # Compare on the shared prefix: identical category indices mean the label
        # is unchanged, since the full alphabet is a prefix extension.
        changed_rows = (old_labels != new_labels).any(axis=1)
        entry["labels_changed"] = int(changed_rows.sum())
        entry["labels_changed_pct"] = 100.0 * float(changed_rows.mean())
        entry["changed_equals_rejected"] = bool(
            set(np.nonzero(changed_rows)[0].tolist()) == set(rejected))
        entry["mean_nodes_differing_on_changed"] = (
            float((old_labels != new_labels)[changed_rows].sum(axis=1).mean())
            if changed_rows.any() else 0.0)

        # Round trip must be exact under the full alphabet.
        rt = (node_preds_to_edge_preds(
            onehot(torch.from_numpy(new_labels), fvr), fi, fo, el.shape[1]) > 0.5
        ).numpy().astype(int)
        entry["roundtrip_exact_pct"] = 100.0 * float((rt == el).all(axis=1).mean())

    report.append(entry)
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--grids", type=int, nargs="+", default=[12, 30])
    ap.add_argument("--splits", nargs="+", default=["test", "train"])
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    print("=== structural checks ===")
    for N in args.grids:
        ok, why = check_legacy_regression(N)
        print(f"  {N}x{N} monotone == pre-fix encoding: {ok}" + ("" if ok else f"  ({why})"))
        assert ok, why
        ok, why = check_prefix_extension(N)
        print(f"  {N}x{N} monotone is prefix of full:   {ok}" + ("" if ok else f"  ({why})"))
        assert ok, why

    print("\n=== alphabet sizes ===")
    sizes = {}
    for N in args.grids:
        row = {}
        for a in ("monotone", "full"):
            _, vr, no, td = build_node_categories(N, alphabet=a)
            row[a] = {"total_dim": td, "n_nodes": len(no),
                      "dist": {str(k): v for k, v in sorted(Counter(vr).items())}}
            print(f"  {N}x{N} {a:<9} total_dim={td:<6} dist={sorted(Counter(vr).items())}")
        sizes[str(N)] = row

    print("\n=== label re-conversion ===")
    report = []
    for N in args.grids:
        for split in args.splits:
            try:
                e = run_split(N, split, args.data_root, report)
            except FileNotFoundError:
                print(f"  {N}x{N} {split}: data missing, skipped")
                continue
            print(f"  {N}x{N} {split:<5} n={e['n']:<6} "
                  f"rejected_by_old_alphabet={e['rejected_by_monotone_alphabet']:<5} "
                  f"({e['rejected_pct']:.2f}%)  full_failures={e['full_alphabet_failures']}  "
                  f"labels_changed={e.get('labels_changed')} "
                  f"(match_rejected={e.get('changed_equals_rejected')})  "
                  f"roundtrip={e.get('roundtrip_exact_pct')}%")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(
            {"sizes": sizes, "splits": report}, indent=2))
        print(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    main()
